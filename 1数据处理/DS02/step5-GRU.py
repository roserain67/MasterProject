import torch
import torch.nn as nn
import numpy as np

# 1. 加载序列数据
seq_path = "feature_all/unit14/feature_selected/sequences_complete.npy"
X = np.load(seq_path)   # 形状 (47, 30, 172)

# 转为 tensor
X_tensor = torch.tensor(X, dtype=torch.float32)   # shape: (47, 30, 172)

# 2. 定义 GRU 编码器
class GRUEncoder(nn.Module):
    def __init__(self, input_dim=172, hidden_dim=128, num_layers=1, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        # 输出 embedding（可作为 state）
        self.output_layer = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_dim)
        """
        output, h_n = self.gru(x)
        # h_n[-1] shape: (batch, hidden_dim) —— 最后一层 GRU 的 hidden state
        emb = self.output_layer(h_n[-1])
        return emb  # shape: (batch, 64)

# 3. 初始化模型
encoder = GRUEncoder(input_dim=172, hidden_dim=128, num_layers=2).cuda()

# 4. 前向传播 -> 得到最终 state 表征
X_batch = X_tensor.cuda()

state_embeddings = encoder(X_batch)
# 保存训练好的 state embeddings

print("输入 X shape:", X_batch.shape)
print("输出 state shape:", state_embeddings.shape)
