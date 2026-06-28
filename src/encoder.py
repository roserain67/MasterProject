import os
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUEncoder(nn.Module):
    def __init__(self, input_dim=43, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, 64)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(x)
        emb = self.output_layer(h_n[-1])
        return emb


def load_gru_encoder(pretrain_path=None):
    model = GRUEncoder().to(DEVICE)
    if pretrain_path and os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=DEVICE))
        print(f">> GRU 预训练权重已加载: {pretrain_path}")
    else:
        print(">> GRU 使用随机初始化")
    return model
