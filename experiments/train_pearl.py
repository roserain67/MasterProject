"""PEARL 训练入口"""
import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pearl import train


def main():
    parser = argparse.ArgumentParser(description="PEARL Meta-RL 训练")
    parser.add_argument("--config", type=str, default="configs/pearl_default.yaml")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.log_dir:
        cfg["log_dir"] = args.log_dir
    if args.num_episodes:
        cfg["num_episodes"] = args.num_episodes

    train(cfg)


if __name__ == "__main__":
    main()
