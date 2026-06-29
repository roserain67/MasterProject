"""策略诊断（只读，不训练）

目的：定位 train/test gap 的根因——
  训练 reward +474（stochastic 靠噪声续命），但确定性测试退化成纯 run +6.6。

方法：加载 best_model，用纯 run 把 env 的 pointer 一路推高，
在每个 pointer 水平打印 actor 的 π 与 critic 的 Q。
重点观察「随 pointer 升高，agent 是否把维修概率/价值顶上去」：

  · 若高 pointer 下 Q(修) > Q(run) 但 π 仍偏 run
        → critic 学对了，actor 没跟上（alpha/熵/lr 问题）
  · 若高 pointer 下 Q(修) 始终 < Q(run)
        → critic 没学到状态依赖（demo 覆盖不足 / 状态表示问题）

用法：
  python experiments/diagnose_policy.py --log_dir logs/compare/PEARL_seed0
  python experiments/diagnose_policy.py --log_dir logs/compare/PEARL_seed2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import torch

from src.utils.paths import find_project_root
os.chdir(find_project_root())

from src.pearl import (
    DEVICE, ContextEncoder, Actor, Critic, sample_z,
)
from src.env import MaintenanceEnv
from src.encoder import load_gru_encoder
from src.utils.data_loader import load_sequences

ACT_NAMES = ["run", "修A", "修B", "修AB", "更换"]


def diag_one_unit(seq, unit_id, encoder_model, actor, critic, context_encoder,
                  state_dim, n_actions, context_input_dim, no_gru):
    """纯 run 推进一条序列，逐步打印 π 与 Q。"""
    env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim, no_gru=no_gru)
    s = env.reset()
    L = env.seq_len

    # z 用 zero-context 的 mu（确定性，与测试初始一致）
    zero_ctx = torch.zeros((1, context_input_dim), device=DEVICE)
    with torch.no_grad():
        mu, _ = context_encoder(zero_ctx)
        z = mu  # 确定性 z

    print(f"\n{'='*92}")
    print(f"Unit {unit_id}  seq_len={L}  （纯 run 推进，z=zero-context mu）")
    print(f"{'='*92}")
    print(f"{'t':>3} {'pA':>4} {'pB':>4} | {'argmax':>6} | "
          f"{'π:run':>6}{'修A':>6}{'修B':>6}{'修AB':>6}{'更换':>6} | "
          f"{'Q:run':>7}{'修A':>7}{'修B':>7}{'修AB':>7}{'更换':>7} | {'maxRepairQ-runQ':>15}")

    switched_at = None
    for t in range(L + 5):
        s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            probs = actor(s_t, z).cpu().numpy().flatten()
            q = critic(s_t, z).cpu().numpy().flatten()
        argmax = int(np.argmax(probs))
        repair_gap = float(q[1:].max() - q[0])  # 最优维修动作 Q 减去 run 的 Q
        if argmax != 0 and switched_at is None:
            switched_at = t

        # 只在关键点打印（前几步 + 每 10 步 + pointer 临界附近 + argmax 切换）
        pmax = max(env.pointer_A, env.pointer_B)
        is_key = (t < 3) or (t % 10 == 0) or (pmax >= 0.5 * L) or (argmax != 0)
        if is_key:
            mark = "  <<切到维修" if argmax != 0 else ""
            print(f"{t:>3} {env.pointer_A:>4} {env.pointer_B:>4} | {ACT_NAMES[argmax]:>6} | "
                  f"{probs[0]:>6.3f}{probs[1]:>6.3f}{probs[2]:>6.3f}{probs[3]:>6.3f}{probs[4]:>6.3f} | "
                  f"{q[0]:>7.1f}{q[1]:>7.1f}{q[2]:>7.1f}{q[3]:>7.1f}{q[4]:>7.1f} | "
                  f"{repair_gap:>+15.1f}{mark}")

        # 纯 run 推进
        s, r, done, _ = env.step(0)
        if done:
            print(f"    -> 第 {t} 步 done（故障/终止），停止推进")
            break

    if switched_at is None:
        print(">> 结论：纯 run 全程 argmax 始终是 0，即便 pointer 拉满也不切维修 "
              "—— 确定性策略不会维修。")
    else:
        print(f">> 结论：argmax 在 t={switched_at} 切换到维修动作。")


def main():
    parser = argparse.ArgumentParser(description="策略诊断（只读）")
    parser.add_argument("--config", type=str, default="configs/pearl_default.yaml")
    parser.add_argument("--log_dir", type=str, default="logs/compare/PEARL_seed0")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    state_dim = cfg["state_dim"]
    n_actions = cfg["n_actions"]
    z_dim = cfg["z_dim"]
    context_input_dim = state_dim + 2 + state_dim
    no_gru = cfg.get("no_gru", False)

    encoder_model = None if no_gru else load_gru_encoder(cfg.get("pretrain_path"))

    # 构建网络并加载 best_model
    context_encoder = ContextEncoder(context_input_dim, z_dim=z_dim).to(DEVICE)
    actor = Actor(state_dim, z_dim, n_actions).to(DEVICE)
    critic = Critic(state_dim, z_dim, n_actions).to(DEVICE)

    ckpt_path = os.path.join(args.log_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"未找到 {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    actor.load_state_dict(ckpt["actor"])
    context_encoder.load_state_dict(ckpt["encoder"])
    if "critic" in ckpt:
        critic.load_state_dict(ckpt["critic"])
    else:
        print("!! best_model 里没有 critic（旧 checkpoint），Q 值不可信")
    actor.eval(); critic.eval(); context_encoder.eval()
    print(f"已加载 {ckpt_path}")

    train_sequences, train_unit_ids = load_sequences(cfg["data_base"], cfg["train_units"])

    # 每个 train unit 取第一条序列做诊断
    seen = set()
    for seq, uid in zip(train_sequences, train_unit_ids):
        if uid in seen:
            continue
        seen.add(uid)
        diag_one_unit(seq, uid, encoder_model, actor, critic, context_encoder,
                      state_dim, n_actions, context_input_dim, no_gru)


if __name__ == "__main__":
    main()
