"""基线算法训练入口 (SAC / TD3 / TQC)"""
import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import find_project_root
os.chdir(find_project_root())


def main():
    parser = argparse.ArgumentParser(description="基线算法训练")
    parser.add_argument("--algo", type=str, required=True, choices=["sac", "td3", "tqc"])
    parser.add_argument("--config", type=str, default="configs/baseline_default.yaml")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.log_dir:
        cfg["log_dir"] = args.log_dir
    else:
        cfg["log_dir"] = f"logs/{args.algo}"
    if args.num_episodes:
        cfg["num_episodes"] = args.num_episodes

    if args.algo == "sac":
        from src.baselines.sac import train
    elif args.algo == "td3":
        from src.baselines.td3 import train
    elif args.algo == "tqc":
        from src.baselines.tqc import train

    train(cfg)


if __name__ == "__main__":
    main()
