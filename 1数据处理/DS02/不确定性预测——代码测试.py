import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np

# # 定义 LSTM RUL 预测器
# class LSTMRULPredictor(nn.Module):
#     def __init__(self):
#         super(LSTMRULPredictor, self).__init__()
#         self.lstm = nn.LSTM(input_size=14, hidden_size=64, num_layers=2, batch_first=True)
#         self.fc = nn.Linear(64, 1)
#         self.dropout = nn.Dropout(p=0.05)
#
#     def forward(self, x):
#         x, _ = self.lstm(x)  # LSTM输出
#         x = x[:, -1, :]  # 取最后一个时间步的输出
#         x = self.dropout(x)
#         x = self.fc(x)  # 全连接层
#         return x
#
#
# # 读取数据集
# data = pd.read_csv('N-CMAPSS_DS02_processed_sampled_no_normalization.csv')
#
# # 整体归一化传感器数据
# for column in ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']:
#     mean = data[column].mean()
#     std = data[column].std()
#     data[column] = (data[column] - mean) / std
#
#
# # 加载训练好的模型
# lstm_rul_predictor = LSTMRULPredictor()
# lstm_rul_predictor.load_state_dict(torch.load('lstm_rul_predictor.pth'))
#
# # 用到的工况参数和传感器数据列
# sensor_columns = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
# condition_columns = ['alt', 'Mach', 'TRA', 'T2']
#
# # 进行预测并还原RUL
# def predict_rul(lstm_rul_predictor, data, T=3):
#     lstm_rul_predictor.train()
#     df = pd.DataFrame(columns=['unit', 'rul_real', 'rul_pre_mean', 'rul_pre_std'])
#
#     with torch.no_grad():
#         for unit in data['unit'].unique():
#             unit_data = data[data['unit'] == unit].reset_index(drop=True)
#             unit_data_tensor = torch.tensor(unit_data[sensor_columns].values, dtype=torch.float32).unsqueeze(1)
#             # 执行T次预测
#             # 定义均值和标准差矩阵，用于存储T次预测的结果
#             mean_matrix = np.zeros((len(unit_data), T))
#             for _ in range(T):
#                 prediction = lstm_rul_predictor(unit_data_tensor)
#                 mean_matrix[:, _] = prediction.squeeze().numpy()
#             # 对均值矩阵按行求均值，得到对应unit下每个测试样本的预测均值
#             unit_mean = np.mean(mean_matrix, axis=1) * 100
#             unit_std = np.std(mean_matrix, axis=1) * 100
#             for idx, row in unit_data.iterrows():
#                 df = df._append({'unit': row['unit'], 'rul_real': row['RUL'], 'rul_pre_mean': unit_mean[idx], 'rul_pre_std': unit_std[idx]}, ignore_index=True)
#     return df
#
# # 进行RUL预测
# predictions_df = predict_rul(lstm_rul_predictor, data)
#
# # 保存预测结果
# predictions_df.to_csv('不确定性预测结果.csv', index=False)


# 按照发动机编号分组绘制预测结果
predictions_df = pd.read_csv('不确定性预测结果.csv')
for unit in predictions_df['unit'].unique():
    plt.figure(figsize=(12, 6))
    unit_data = predictions_df[predictions_df['unit'] == unit]
    plt.plot(unit_data.index, unit_data['rul_real'], color='blue', alpha=0.5)
    plt.plot(unit_data.index, unit_data['rul_pre_mean'], color='red', alpha=0.5)
    plt.plot(unit_data.index, unit_data['rul_pre_mean'] + 2 * unit_data['rul_pre_std'], color='red', linestyle='--', alpha=0.5)
    plt.plot(unit_data.index, unit_data['rul_pre_mean'] - 2 * unit_data['rul_pre_std'], color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('RUL')
    plt.title('RUL Prediction vs Actual (Unit {})'.format(unit))
    plt.legend(['Actual RUL', 'Predicted RUL', '95% Confidence Interval'])
    plt.show()

