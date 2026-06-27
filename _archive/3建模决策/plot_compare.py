"""
读取 compare 目录下各模型子目录（SAC、TD3、TQC、PEARL 等）的 loss_reward_action.csv 与 test_results_*.csv，
将不同模型的训练、测试结果绘制在一张图上。
用法：
  python plot_compare.py
  python plot_compare.py --dir ../logs/compare
  python plot_compare.py -d ../logs/compare -o compare_figures.png
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 模型显示名与颜色、线型
MODEL_STYLE = {
    "SAC":   {"color": "#1f77b4", "linestyle": "-", "marker": None},
    "TD3":   {"color": "#ff7f0e", "linestyle": "-", "marker": None},
    "TQC":   {"color": "#2ca02c", "linestyle": "-", "marker": None},
    "PEARL": {"color": "#d62728", "linestyle": "-", "marker": None},
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
    mean = s.rolling(window, min_periods=1).mean()
    std = s.rolling(window, min_periods=1).std().fillna(0)
    return mean.values, std.values


def main():
    parser = argparse.ArgumentParser(description="多模型训练/测试曲线对比图")
    parser.add_argument("-d", "--dir", default=None, help="compare 根目录，默认为 ../logs/compare")
    parser.add_argument("-o", "--output", default=None, help="输出文件名前缀，不指定则保存到 compare 目录下")
    parser.add_argument("--train-window", type=int, default=30, help="训练曲线平滑窗口")
    parser.add_argument("--test-window", type=int, default=20, help="测试曲线平滑窗口")
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    compare_dir = args.dir or os.path.join(base, "..", "logs", "compare")
    if not os.path.isdir(compare_dir):
        print(f"目录不存在: {compare_dir}")
        return

    subdirs = [d for d in os.listdir(compare_dir) if os.path.isdir(os.path.join(compare_dir, d))]
    # 按固定顺序
    order = ["SAC", "TD3", "TQC", "PEARL"]
    model_dirs = [m for m in order if m in subdirs]
    model_dirs += [m for m in subdirs if m not in order]

    # ----- 图1: 训练 reward 对比 -----
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    for name in model_dirs:
        log_dir = os.path.join(compare_dir, name)
        ep, rew = load_train_rewards(log_dir)
        if ep is None or len(ep) == 0:
            continue
        style = MODEL_STYLE.get(name, {"color": None, "linestyle": "-", "marker": None})
        color = style["color"] or plt.cm.tab10(len(ax1.get_lines()) % 10)
        mean, std = smooth_curve(rew, args.train_window)
        ax1.plot(ep, mean, color=color, linestyle=style["linestyle"], label=name, linewidth=2)
        ax1.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("奖励")
    ax1.set_title("训练阶段 Episode Reward 对比（均值 ± 标准差）")
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    out1 = args.output
    if not out1:
        out1 = os.path.join(compare_dir, "compare_train_reward.png")
    else:
        out1 = out1 if out1.endswith(".png") else out1 + "_train.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"已保存: {out1}")

    # ----- 图2: 测试 reward 对比（unit14 / unit15 各一张或同图两条线 per 模型）-----
    for unit_id in [14, 15]:
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
        has_any = False
        for name in model_dirs:
            log_dir = os.path.join(compare_dir, name)
            ep, rew = load_test_rewards(log_dir, unit_id)
            if ep is None or len(ep) == 0:
                continue
            has_any = True
            style = MODEL_STYLE.get(name, {"color": None, "linestyle": "-", "marker": None})
            color = style["color"] or plt.cm.tab10(len(ax2.get_lines()) % 10)
            mean, std = smooth_curve(rew, args.test_window)
            ax2.plot(ep, mean, color=color, linestyle=style["linestyle"], label=name, linewidth=2)
            ax2.fill_between(ep, mean - std, mean + std, color=color, alpha=0.2)
        if not has_any:
            plt.close(fig2)
            continue
        ax2.set_xlabel("Test Episode")
        ax2.set_ylabel("奖励")
        ax2.set_title(f"测试阶段 Reward 对比（Unit {unit_id}）")
        ax2.legend(loc="best", fontsize=10)
        ax2.grid(True, alpha=0.3)
        out2 = args.output
        if not out2:
            out2 = os.path.join(compare_dir, f"compare_test_reward_unit{unit_id}.png")
        else:
            out2 = out2.replace(".png", "") + f"_test_unit{unit_id}.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"已保存: {out2}")

    # ----- 图3: 训练+测试 合在一张图（子图）-----
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
    for name in model_dirs:
        log_dir = os.path.join(compare_dir, name)
        style = MODEL_STYLE.get(name, {"color": None, "linestyle": "-"})
        color = style["color"] or plt.cm.tab10(model_dirs.index(name) % 10)
        ep, rew = load_train_rewards(log_dir)
        if ep is not None and len(ep) > 0:
            mean, std = smooth_curve(rew, args.train_window)
            ax3a.plot(ep, mean, color=color, linestyle=style["linestyle"], label=name, linewidth=1.5)
            ax3a.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15)
        ep, rew = load_test_rewards(log_dir, 14)
        if ep is not None and len(ep) > 0:
            mean, std = smooth_curve(rew, args.test_window)
            ax3b.plot(ep, mean, color=color, linestyle=style["linestyle"], label=name, linewidth=1.5)
            ax3b.fill_between(ep, mean - std, mean + std, color=color, alpha=0.15)
    ax3a.set_xlabel("Episode")
    ax3a.set_ylabel("奖励")
    ax3a.set_title("训练 Reward")
    ax3a.legend(loc="best", fontsize=9)
    ax3a.grid(True, alpha=0.3)
    ax3b.set_xlabel("Test Episode")
    ax3b.set_ylabel("奖励")
    ax3b.set_title("测试 Reward (Unit 14)")
    ax3b.legend(loc="best", fontsize=9)
    ax3b.grid(True, alpha=0.3)
    out3 = os.path.join(compare_dir, "compare_train_and_test.png")
    if args.output:
        out3 = args.output.replace(".png", "") + "_train_and_test.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"已保存: {out3}")


if __name__ == "__main__":
    main()
