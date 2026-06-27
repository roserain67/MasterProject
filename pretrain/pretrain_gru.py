"""
GRU 预训练：学习编码健康/退化阶段
任务：给定序列前缀，预测归一化位置 position/seq_len（0=健康，1=退化）
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import GRUEncoder
from src.utils.data_loader import load_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_pretrain_samples(sequences, min_len=5):
    samples = []
    for seq in sequences:
        T = len(seq)
        if T < min_len:
            continue
        for t in range(min_len, T):
            prefix = seq[: t + 1].astype(np.float32)
            target = t / max(1, T - 1)
            samples.append((prefix, target))
    return samples


def train(data_base, unit_ids, save_path, epochs=100, batch_size=32):
    sequences, _ = load_sequences(data_base, unit_ids)
    if len(sequences) == 0:
        raise ValueError(f"未找到序列数据: {data_base}")

    samples = create_pretrain_samples(sequences)
    print(f"预训练样本数: {len(samples)}")

    model = GRUEncoder().to(DEVICE)
    regressor = nn.Linear(64, 1).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=1e-3)

    for ep in range(epochs):
        np.random.shuffle(samples)
        total_loss = 0
        n_batches = 0
        for i in range(0, len(samples), batch_size):
            batch = samples[i: i + batch_size]
            prefixes = [s[0] for s in batch]
            targets = torch.tensor([s[1] for s in batch], dtype=torch.float32, device=DEVICE)

            max_len = max(p.shape[0] for p in prefixes)
            padded = []
            for p in prefixes:
                if p.shape[0] < max_len:
                    pad = np.zeros((max_len - p.shape[0], p.shape[1]), dtype=np.float32)
                    p = np.concatenate([p, pad], axis=0)
                padded.append(p)
            x = torch.tensor(np.stack(padded), dtype=torch.float32, device=DEVICE)

            emb = model(x)
            pred = regressor(emb).squeeze(-1)
            loss = nn.functional.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}: loss={total_loss/max(1,n_batches):.6f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"GRU 已保存至 {save_path}")


def main():
    parser = argparse.ArgumentParser(description="GRU 预训练")
    parser.add_argument("--data_base", type=str, default="1数据处理/DS02/feature_all")
    parser.add_argument("--units", type=int, nargs="+", default=[14, 16, 18, 20])
    parser.add_argument("--save_path", type=str, default="pretrain/gru_pretrained.pt")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    train(args.data_base, args.units, args.save_path, args.epochs)


if __name__ == "__main__":
    main()
