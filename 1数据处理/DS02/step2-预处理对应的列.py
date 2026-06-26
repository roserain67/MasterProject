import numpy as np
import pandas as pd
import h5py

# 设置文件位置
import pandas as pd
import h5py

# 设置文件位置
filename = r'D:\yyo-Python\0毕设\数据处理\DS02\data\N-CMAPSS_DS02-006.h5'

# 读取数据
with h5py.File(filename, 'r') as hdf:
    # Load subsets of data
    A_dev = pd.DataFrame(hdf['A_dev'][:], columns=[x.decode() for x in hdf['A_var'][:]])
    W_dev = pd.DataFrame(hdf['W_dev'][:], columns=[x.decode() for x in hdf['W_var'][:]]) # W,工况 4列，W: alt、Mach、TRA、T2
    X_s_dev = pd.DataFrame(hdf['X_s_dev'][:], columns=[x.decode() for x in hdf['X_s_var'][:]]) # X_s，测量信号,14列，选择 T24、T48、Nf、Nc、Wf
    X_v_dev = pd.DataFrame(hdf['X_v_dev'][:], columns=[x.decode() for x in hdf['X_v_var'][:]])  # X_v，虚拟信号，14列，选择 P30
    T_dev = pd.DataFrame(hdf['T_dev'][:], columns=[x.decode() for x in hdf['T_var'][:]])   # 10列，健康参数,选择：HPT_eff_mod, LPT_flow_mod, LPT_eff_mod

    A_test = pd.DataFrame(hdf['A_test'][:], columns=[x.decode() for x in hdf['A_var'][:]])
    W_test = pd.DataFrame(hdf['W_test'][:], columns=[x.decode() for x in hdf['W_var'][:]])
    X_s_test = pd.DataFrame(hdf['X_s_test'][:], columns=[x.decode() for x in hdf['X_s_var'][:]])
    X_v_test = pd.DataFrame(hdf['X_v_test'][:], columns=[x.decode() for x in hdf['X_v_var'][:]])
    T_test = pd.DataFrame(hdf['T_test'][:], columns=[x.decode() for x in hdf['T_var'][:]])

# Concatenate development and test sets
A = pd.concat([A_dev, A_test], ignore_index=True)
W = pd.concat([W_dev, W_test], ignore_index=True)
X_s = pd.concat([X_s_dev, X_s_test], ignore_index=True)
X_v = pd.concat([X_v_dev, X_v_test], ignore_index=True)
T = pd.concat([T_dev, T_test], ignore_index=True)
combined_df = pd.concat([A, W, X_s, X_v, T], axis=1)
combined_df.to_csv('combined_data.csv', index=False)

print('Combined data saved to combined_data.csv')


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# # 读取原始数据
data = pd.read_csv('combined_data.csv')
# print(data.shape)
# # 打印data的列名
# print(data.columns)
# # 选择需要的列
selected_columns = ['unit', 'cycle', 'alt', 'Mach', 'TRA', 'T2', 'Nf' , 'Nc' , 'T24', 'T48' , 'P30' , 'Wf' , 'HPT_eff_mod', 'LPT_flow_mod', 'LPT_eff_mod']
data_need = data[selected_columns]
data_need.to_csv('N-CMAPSS_DS02_selected.csv', index=False)


data = pd.read_csv('N-CMAPSS_DS02_selected.csv')
# 对6个传感器(Nf, Nc, T24, T48, P30, Wf)用4个工况量拟合，取残差

# 下采样

def downsample(df, rate=10):
    return df.iloc[::rate, :]

# 归一化
def normalize(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler

# 滑动窗口处理
def sliding_window(df, window_size=50, step=25):
    samples = []
    for unit in df['unit'].unique():
        unit_data = df[df['unit'] == unit]
        for start in range(0, len(unit_data) - window_size + 1, step):
            end = start + window_size
            sample = unit_data.iloc[start:end]
            samples.append(sample)
    return samples

# 提取操作条件参数和传感器列名
condition_columns = ['alt', 'Mach', 'TRA', 'T2']
sensor_columns =['Nf' , 'Nc' , 'T24', 'T48' , 'P30' , 'Wf']
health_columns = ['HPT_eff_mod', 'LPT_flow_mod', 'LPT_eff_mod']
feature_columns = condition_columns + sensor_columns + health_columns

# 数据预处理
# 1. 下采样
data_downsampled = downsample(data)

# 2. 归一化
data_normalized, scaler = normalize(data_downsampled, feature_columns)

# 3. 滑动窗口处理
samples = sliding_window(data_normalized)

# 将处理后的样本转换为DataFrame格式
processed_data = pd.concat(samples, ignore_index=True)

# 保存处理后的数据集
processed_data.to_csv('N-CMAPSS_DS02_processed.csv', index=False)
