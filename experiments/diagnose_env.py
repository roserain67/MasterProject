"""
环境诊断脚本（只读环境，不训练）
=================================
目的：验证 MaintenanceEnv 的 reward 结构本身是否合理，回答三个问题：
  1) 最优策略到底能拿多少分？是否真的可达 +600 量级？
  2) reward shaping (shaping_k) 是否在扭曲激励，鼓励维修/更换 spam？
  3) 能否复现训练日志里的坍缩值（PEARL 修A×10≈-164, SAC 换×10≈-467）？

用法：
  python experiments/diagnose_env.py

拆解方法（无假设）：
  每步 r = run(+3) + cost + shaping + terminal
  run / shaping / terminal 都能从环境的可观测量精确算出，
  因此 cost = r - run - shaping - terminal 是精确反推，不复制任何成本公式。
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import find_project_root
os.chdir(find_project_root())

import numpy as np
from src.env import MaintenanceEnv
from src.utils.data_loader import load_sequences

DATA = "1数据处理/DS02/feature_all/unified"
ACTION_NAMES = {0: "run", 1: "修A", 2: "修B", 3: "修AB", 4: "更换"}


def decompose_step(env, action):
    """执行一步并把 reward 精确拆解为 run/cost/shaping/terminal 四项。"""
    phi_old = env._potential()
    _, r, done, _ = env.step(action)
    phi_new = env._potential()

    run = env.R_run
    shaping = env.gamma * phi_new - phi_old

    terminal = 0.0
    if done:
        if env.repair_in_row >= env.max_repair_in_row:
            terminal += -env.penalty_over_repair
        if env.pointer_A >= env.seq_len - 1 or env.pointer_B >= env.seq_len - 1:
            terminal += -env.penalty_break
        elif env.current_step >= env.max_steps:
            terminal += env.survival_bonus

    cost = r - run - shaping - terminal  # 精确反推
    return r, done, dict(run=run, cost=cost, shaping=shaping, terminal=terminal)


def run_policy(seq, policy_fn, shaping_k=None, max_steps=200):
    env = MaintenanceEnv(seq, None, state_dim=2, no_gru=True, max_steps=max_steps)
    if shaping_k is not None:
        env.shaping_k = shaping_k
    env.reset()
    total = 0.0
    comp = dict(run=0.0, cost=0.0, shaping=0.0, terminal=0.0)
    acts = []
    term_cause = "max_len"
    for t in range(max_steps):
        a = policy_fn(env, t)
        r, done, parts = decompose_step(env, a)
        total += r
        for k in comp:
            comp[k] += parts[k]
        acts.append(a)
        if done:
            if env.repair_in_row >= env.max_repair_in_row:
                term_cause = "over_repair"
            elif env.pointer_A >= env.seq_len - 1 or env.pointer_B >= env.seq_len - 1:
                term_cause = "break(故障)"
            elif env.current_step >= env.max_steps:
                term_cause = "survival(存活)"
            break
    hist = {ACTION_NAMES[i]: acts.count(i) for i in range(5) if acts.count(i) > 0}
    return dict(total=total, length=len(acts), term=term_cause, comp=comp, hist=hist)


# ---------- 手写策略 ----------
def p_run(env, t):       return 0
def p_repairA(env, t):   return 1          # 复现 PEARL 坍缩
def p_replace(env, t):   return 4          # 复现 SAC 坍缩


def make_threshold(thr):
    """阈值策略：哪个部件接近故障线就修哪个，否则 run。修完会自动回落 → 下一步 run，避免连修。"""
    def pol(env, t):
        L = env.seq_len
        a_hot = env.pointer_A >= thr * L
        b_hot = env.pointer_B >= thr * L
        if a_hot and b_hot:
            return 3
        if a_hot:
            return 1
        if b_hot:
            return 2
        return 0
    return pol


def fmt(res):
    c = res["comp"]
    return (f"total={res['total']:8.1f} | len={res['length']:3d} | {res['term']:12s} | "
            f"run={c['run']:7.1f} cost={c['cost']:8.1f} shape={c['shaping']:8.1f} term={c['terminal']:7.1f} | {res['hist']}")


def main():
    seqs, ids = load_sequences(DATA, [16, 18, 20])
    if not seqs:
        print(f"!! 没找到数据: {DATA}")
        return
    print(f"加载 {len(seqs)} 条训练序列, seq_len = {[s.shape[0] for s in seqs]}\n")
    seq = seqs[0]
    L = seq.shape[0]

    print("=" * 110)
    print(f"实验A: 默认 shaping_k=60，各策略表现（seq_len={L}）")
    print("=" * 110)
    base = {
        "纯run(不修)":      p_run,
        "修A×spam(PEARL)":  p_repairA,
        "更换×spam(SAC)":   p_replace,
    }
    for name, pol in base.items():
        print(f"{name:18s} {fmt(run_policy(seq, pol))}")
    print("  -- 阈值策略 (找最优阈值) --")
    best = None
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        res = run_policy(seq, make_threshold(thr))
        print(f"  阈值={thr:.1f}        {fmt(res)}")
        if best is None or res["total"] > best[1]["total"]:
            best = (thr, res)
    print(f"\n  >> 最优阈值策略: thr={best[0]}, total={best[1]['total']:.1f}  "
          f"(理论最优≈ 200*3+50-维修成本 ≈ +600)")

    print("\n" + "=" * 110)
    print("实验B: shaping_k 扫描 —— 看 shaping 是否扭曲激励 (改变策略排名)")
    print("=" * 110)
    print(f"{'策略':18s} {'k=0':>12s} {'k=5':>12s} {'k=30':>12s} {'k=60':>12s}")
    pols = {"纯run": p_run, "修A×spam": p_repairA, "更换×spam": p_replace,
            "阈值0.7": make_threshold(0.7)}
    for name, pol in pols.items():
        row = [run_policy(seq, pol, shaping_k=k)["total"] for k in (0, 5, 30, 60)]
        print(f"{name:18s} " + " ".join(f"{v:12.1f}" for v in row))
    print("  解读: 若 k=0 时'阈值'最高、但 k=60 时'spam'反超 → shaping 在鼓励 spam。")

    print("\n" + "=" * 110)
    print("实验C: 单步奖励尺度 —— shaping 尖峰 vs 真实信号(run=3, cost≈3~16)")
    print("=" * 110)
    env = MaintenanceEnv(seq, None, state_dim=2, no_gru=True)
    env.reset()
    print(f"{'step':>4s} {'action':>6s} {'run':>6s} {'cost':>8s} {'shaping':>9s} {'r_total':>9s} {'pA':>4s} {'pB':>4s}")
    pol = make_threshold(0.7)
    for t in range(30):
        a = pol(env, t)
        r, done, parts = decompose_step(env, a)
        print(f"{t:>4d} {ACTION_NAMES[a]:>6s} {parts['run']:>6.1f} {parts['cost']:>8.1f} "
              f"{parts['shaping']:>9.1f} {r:>9.1f} {env.pointer_A:>4d} {env.pointer_B:>4d}")
        if done:
            print("  (terminated)")
            break


if __name__ == "__main__":
    main()
