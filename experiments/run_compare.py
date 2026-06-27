"""多算法对比运行脚本"""
import argparse
import yaml
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import load_gru_encoder
from src.utils.data_loader import load_sequences


def main():
    parser = argparse.ArgumentParser(description="多算法对比训练")
    parser.add_argument("--config", type=str, default="configs/baseline_default.yaml")
    parser.add_argument("--log_base", type=str, default="logs/compare")
    parser.add_argument("--num_episodes", type=int, default=300)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    encoder_model = load_gru_encoder(cfg.get("pretrain_path"))
    train_sequences, _ = load_sequences(cfg["data_base"], cfg["train_units"])
    if not train_sequences:
        raise ValueError("未找到训练数据")

    test_by_unit = {}
    for uid in cfg.get("test_units", [14, 15]):
        seqs, _ = load_sequences(cfg["data_base"], [uid])
        if seqs:
            test_by_unit[uid] = seqs

    cfg["num_episodes"] = args.num_episodes

    from src.baselines.sac import train as train_sac
    from src.baselines.td3 import train as train_td3
    from src.baselines.tqc import train as train_tqc

    for algo_name, train_fn in [("SAC", train_sac), ("TD3", train_td3), ("TQC", train_tqc)]:
        algo_cfg = dict(cfg)
        algo_cfg["log_dir"] = os.path.join(args.log_base, algo_name)
        print(f"\n{'='*50}")
        print(f"开始训练: {algo_name}")
        print(f"{'='*50}")
        train_fn(algo_cfg, encoder_model=encoder_model,
                 train_sequences=train_sequences, test_by_unit=test_by_unit)


if __name__ == "__main__":
    main()
