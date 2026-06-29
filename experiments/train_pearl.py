"""PEARL 训练入口"""
import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import find_project_root
os.chdir(find_project_root())

from src.pearl import train


def main():
    parser = argparse.ArgumentParser(description="PEARL Meta-RL 训练")
    parser.add_argument("--config", type=str, default="configs/pearl_default.yaml")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.log_dir:
        cfg["log_dir"] = args.log_dir
    if args.num_episodes:
        cfg["num_episodes"] = args.num_episodes
    if args.seed is not None:
        cfg["seed"] = args.seed
        # 没有显式 log_dir 时，每个种子写到独立目录，避免 diagnostics.csv 互相覆盖
        if not args.log_dir:
            cfg["log_dir"] = cfg["log_dir"].rstrip("/") + f"_seed{args.seed}"

    train(cfg)


if __name__ == "__main__":
    main()
