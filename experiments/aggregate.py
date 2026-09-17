"""跨方法/跨种子汇总，并把指标换成可跨 unit 比较的口径。

为什么不能直接比 reward（plan.md P4/P5）：
  1. reward 含 potential-based shaping。shaping 保证最优策略不变，但**不保证整集回报
     不变** —— 修AB 策略整集 shaping ≈ +23，更换策略 ≈ +1.5。直接比 reward 会把
     20 多分的 shaping 差算成经济收益。→ 一律用 reward_econ。
  2. 不同 unit 的寿命不同（seq_len 63~76），200 步内需要的维修次数就不同
     （unit14 修 2 次、unit16 修 4 次），reward 的 unit 间差异主要由寿命决定，
     不是泛化好坏。"测试 unit 比训练 unit 分高"是寿命假象。
     → 报"距规则策略上界的缺口"和"达成率 = 经济回报 / 上界"。

用法：
  python experiments/aggregate.py --log_base logs/final
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.utils.paths import find_project_root
os.chdir(find_project_root())

from src.env import MaintenanceEnv
from src.pearl import _threshold_policy
from src.utils.data_loader import load_sequences

DATA_BASE = "1数据处理/DS02/feature_all/unified"


def _rollout(seq, policy_fn, max_steps=200):
    env = MaintenanceEnv(seq, None, state_dim=2, no_gru=True)
    env.reset()
    tot = econ = 0.0
    for _ in range(max_steps):
        _, r, done, info = env.step(policy_fn(env))
        tot += r
        econ += info["reward_econ"]
        if done:
            break
    return tot, econ


def reference_policies(seq):
    """规则策略参考线：给出问题的上下界，所有学习方法都拿它们做标尺。"""
    refs = {}
    refs["事后维修"] = _rollout(seq, lambda e: 0)
    for name, ra in [("阈值-修AB", 3), ("阈值-更换", 4)]:
        best = (-np.inf, None, None)
        for thr in np.arange(0.30, 0.99, 0.01):
            tot, econ = _rollout(seq, lambda e, t=thr, a=ra: _threshold_policy(e, t, repair_action=a))
            if econ > best[0]:
                best = (econ, tot, float(thr))
        refs[name] = (best[1], best[0])
        refs[name + "_thr"] = best[2]
    return refs


def main():
    parser = argparse.ArgumentParser(description="跨方法汇总 + 归一化指标")
    parser.add_argument("--log_base", type=str, default="logs/final")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    out_path = args.out or os.path.join(args.log_base, "compare_summary.csv")

    # ---------- 规则策略参考线 ----------
    units = sorted({int(os.path.basename(p).split("_")[1].replace("unit", ""))
                    for p in glob.glob(os.path.join(args.log_base, "*", "eval_unit*_K*_summary.csv"))})
    refs = {}
    print("规则策略参考线（经济回报口径，已剔除 shaping）")
    print(f"{'unit':>5}{'L':>5}{'事后维修':>10}{'阈值-更换':>11}{'thr':>6}{'阈值-修AB(上界)':>16}{'thr':>6}")
    for u in units:
        seqs, _ = load_sequences(DATA_BASE, [u])
        if not seqs:
            continue
        r = reference_policies(seqs[0])
        refs[u] = r
        print(f"{u:>5}{seqs[0].shape[0]:>5}{r['事后维修'][1]:>10.1f}"
              f"{r['阈值-更换'][1]:>11.1f}{r['阈值-更换_thr']:>6.2f}"
              f"{r['阈值-修AB'][1]:>16.1f}{r['阈值-修AB_thr']:>6.2f}")

    # ---------- 扫描学习方法的评测结果 ----------
    rows = []
    for path in sorted(glob.glob(os.path.join(args.log_base, "*", "eval_unit*_K*_summary.csv"))):
        run = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path)
        unit = int(fname.split("_")[1].replace("unit", ""))
        k = int(fname.split("_")[2].replace("K", ""))
        df = pd.read_csv(path)
        if "reward_econ" not in df.columns:
            print(f"!! {path} 没有 reward_econ 列（评测口径改动前生成的），跳过。"
                  f" 先跑 experiments/reevaluate.py --log_dir {os.path.dirname(path)}")
            continue
        upper = refs.get(unit, {}).get("阈值-修AB", (np.nan, np.nan))[1]
        act_cols = [c for c in df.columns if c.startswith("n_act")]
        rows.append({
            "run": run, "unit": unit, "K": k, "n_ep": len(df),
            "reward": df["reward"].mean(),
            "econ": df["reward_econ"].mean(),
            "econ_std": df["reward_econ"].std(),
            "上界": upper,
            "缺口": df["reward_econ"].mean() - upper,
            "达成率": df["reward_econ"].mean() / upper if upper else np.nan,
            "存活率": (df["reason"] == "survive").mean(),
            **{c: df[c].mean() for c in act_cols},
        })

    if not rows:
        print("\n没有可汇总的结果。")
        return

    out = pd.DataFrame(rows).sort_values(["run", "K", "unit"])
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n各 run × unit 的经济回报与达成率（K=0）")
    print(f"{'run':<22}{'unit':>5}{'经济回报':>10}{'上界':>9}{'缺口':>8}{'达成率':>8}{'存活率':>8}")
    for _, r in out[out.K == 0].iterrows():
        print(f"{r['run']:<22}{int(r['unit']):>5}{r['econ']:>10.1f}{r['上界']:>9.1f}"
              f"{r['缺口']:>+8.1f}{r['达成率']:>8.1%}{r['存活率']:>8.0%}")

    print(f"\n按 run 汇总（K=0，跨 unit 平均达成率才是可比的泛化指标）")
    g = out[out.K == 0].groupby("run").agg(
        平均经济回报=("econ", "mean"), 平均缺口=("缺口", "mean"),
        平均达成率=("达成率", "mean"), 存活率=("存活率", "mean"))
    print(g.to_string(float_format=lambda v: f"{v:8.3f}"))
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
