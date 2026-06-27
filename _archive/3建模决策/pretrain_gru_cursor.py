"""
GRU 预训练：学习编码健康/退化阶段
任务：给定序列前缀，预测归一化位置 position/seq_len（0=健康，1=退化）
预训练后 GRU 的 state 将携带健康语义，便于策略学习「健康→运行，退化→维修」
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = os.path.join(os.path.dirname(__file__), "gru_pretrained.pt")


class GRUEncoder(nn.Module):
    def __init__(self, input_dim=172, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        output, h_n = self.gru(x)
        emb = self.output_layer(h_n[-1])
        return emb  # (batch, 64)


def load_sequences(data_base_path, unit_ids, use_trajectory=True):
    all_sequences = []
    for unit_id in unit_ids:
        traj_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "trajectory_complete.npy")
        seq_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "sequences_complete.npy")
        if use_trajectory and os.path.exists(traj_path):
            traj = np.load(traj_path)
            all_sequences.append(traj)
        elif os.path.exists(seq_path):
            seqs = np.load(seq_path)
            for i in range(len(seqs)):
                all_sequences.append(seqs[i])
    return all_sequences


def create_pretrain_samples(sequences, min_len=5):
    """
    为每个序列的每个位置 t 创建样本：(prefix[:t+1], target=t/len)
    """
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


def train():
    data_base = os.path.join(os.path.dirname(__file__), "..", "1数据处理", "DS02", "feature_all")
    if not os.path.exists(data_base):
        data_base = os.path.join("1数据处理", "DS02", "feature_all")

    sequences = load_sequences(data_base, [14, 16, 18, 20])
    if len(sequences) == 0:
        raise ValueError("未找到序列数据。请先运行 step4.2 生成 trajectory_complete.npy 或 sequences_complete.npy")

    samples = create_pretrain_samples(sequences)
    print(f"预训练样本数: {len(samples)}")

    model = GRUEncoder().to(DEVICE)
    regressor = nn.Linear(64, 1).to(DEVICE)
    optimizer = optim.Adam(list(model.parameters()) + list(regressor.parameters()), lr=1e-3)

    EPOCHS = 100
    BATCH = 32

    for ep in range(EPOCHS):
        np.random.shuffle(samples)
        total_loss = 0
        n_batches = 0
        for i in range(0, len(samples), BATCH):
            batch = samples[i : i + BATCH]
            prefixes = [s[0] for s in batch]
            targets = torch.tensor([s[1] for s in batch], dtype=torch.float32, device=DEVICE)

            # Pad to same length
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

    # 保存 GRU（不含 regressor，供 pearl 使用）
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"GRU 已保存至 {SAVE_PATH}")


if __name__ == "__main__":
    train()
