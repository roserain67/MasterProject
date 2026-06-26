import pandas as pd
import numpy as np

# 读取处理过的传感器数据
data = pd.read_csv('N-CMAPSS_DS02_final.csv')

# 选择传感器列
sensor_columns = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24', 'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']
# 选择工况条件列
condition_columns = ['alt', 'Mach', 'TRA', 'T2']  # 确保这些是实际的工况条件列名

# 创建一个空的列表用于存储处理后的数据
processed_data = []

# 获取所有发动机编号
units = data['unit'].unique()

# 设置每个周期取100条样本
samples_per_cycle = 100

# 遍历每个发动机
for unit in units:
    unit_data = data[data['unit'] == unit].sort_values(by='cycle')
    # 获取所有周期编号
    cycles = unit_data['cycle'].unique()

    # 遍历每个周期
    for cycle in cycles:
        cycle_data = unit_data[unit_data['cycle'] == cycle]

        # 确保每个周期取100条数据（或少于100条时取所有数据）
        if len(cycle_data) > samples_per_cycle:
            sampled_data = cycle_data.sample(n=samples_per_cycle, random_state=42)
        else:
            sampled_data = cycle_data

        # 将采样后的数据加入到结果中
        for _, row in sampled_data.iterrows():
            processed_row = [row['unit'], row['cycle']] + row[sensor_columns].tolist() + row[
                condition_columns].tolist() + [row['RUL']]
            processed_data.append(processed_row)

# 创建新的DataFrame用于保存
columns = ['unit', 'cycle'] + sensor_columns + condition_columns + ['RUL']
processed_df = pd.DataFrame(processed_data, columns=columns)

# 保存到新的CSV文件
processed_df.to_csv('N-CMAPSS_DS02_processed_sampled_no_normalization.csv', index=False)

print("已生成未归一化的测试数据，保存为 'N-CMAPSS_DS02_processed_sampled_no_normalization.csv'")
