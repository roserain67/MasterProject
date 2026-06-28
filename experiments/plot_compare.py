"""
多模型训练/测试曲线对比图
用法：
  python experiments/plot_compare.py
  python experiments/plot_compare.py --dir logs/compare
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import find_project_root
os.chdir(find_project_root())

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_STYLE = {
    "SAC":   {"color": "#1f77b4", "linestyle": "-"},
    "TD3":   {"color": "#ff7f0e", "linestyle": "-"},
    "TQC":   {"color": "#2ca02c", "linestyle": "-"},
    "PEARL": {"color": "#d62728", "linestyle": "-"},
}


def load_train_rewards(log_dir):
    path = os.path.join(log_dir, "loss_reward_action.csv")
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    if "reward" not in df.columns:
        return None, None
    ep = df["episode"].values if "episode" in df.columns else np.arange(len(df))
    return ep, df["reward"].values


def load_test_rewards(log_dir, unit_id=14):
    path = os.path.join(log_dir, f"test_results_unit{unit_id}.csv")
    if not os.path.isfile(path):
        return None, None
    df = pd.read_csv(path)
    if "reward" not in df.columns:
        return None, None
    ep = df["episode"].values if "episode" in df.columns else np.arange(len(df))
    return ep, df["reward"].values


def smooth_curve(y, window=20):
    if len(y) < window:
        return y, np.zeros_like(y)
    s = pd.Series(y)
    return s.rolling(window, min_periods=1).mean().values, s.rolling(window, min_periods=1).std().fillna(0).values


def main():
    parser = argparse.ArgumentParser(description="多模型对比图")
    parser.add_argument("-d", "--dir", default="logs/compare")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--train-window", type=int, default=30)
    parser.add_argument("--test-window", type=int, default=20)
    args = parser.parse_args()

    compare_dir = args.dir
    if not os.path.isdir(compare_dir):
        print(f"目录不存在: {compare_dir}")
        return

    subdirs = [d for d in os.listdir(compare_dir) if os.path.isdir(os.path.join(compare_dir, d))]
    order = ["SAC", "TD3", "TQC", "PEARL"]
    model_dirs = [m for m in order if m in subdirs] + [m for m in subdirs if m not in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in model_dirs:
        log_dir = os.path.join(compare_dir, name)
        style = MODEL_STYLE.get(name, {"color": None, "linestyle": "-"})
        color = style["color"] or plt.cm.tab10(model_dirs.index(name) % 10)

        ep, rew = load_train_rewards(log_dir)
        if ep is not None:
            mean, std = smooth_curve(rew, args.train_window)
            ax1.plot(ep, mean, color=color, linestyle=style["linestyle"], label=name, linewidth=2)
            ax1.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2)

        ep, rew = load_test_rewards(log_dir, 14)
        if ep is not None:
            mean, std = smooth_curve(rew, args.test_window)
            ax2.plot(ep, mean, color=color, linestyle=style["linestyle"], label=name, linewidth=2)
            ax2.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("奖励")
    ax1.set_title("训练 Reward 对比")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel("Test Episode")
    ax2.set_ylabel("奖励")
    ax2.set_title("测试 Reward 对比 (Unit 14)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    out = args.output or os.path.join(compare_dir, "compare_all.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")


if __name__ == "__main__":
    main()
