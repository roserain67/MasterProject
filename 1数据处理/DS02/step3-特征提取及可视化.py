'''
工况：
传感器参数：
3个健康表征：取一个定值

'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression

# === 参数区 ===
file_path = 'D:\\yyo-Python\\0毕设\\数据处理\\DS02\\N-CMAPSS_DS02_selected.csv'  # 输入文件路径
save_dir = 'feature_all\\unit14'  # 输出文件夹名称
os.makedirs(save_dir, exist_ok=True)

# === 1. 读取数据 ===
df = pd.read_csv(file_path)

# 检查列结构
print("列名:", df.columns.tolist())

# 假设格式为: unit, cycle, sensor1, sensor2, ..., 工况参数
unit_col = df.columns[0]
cycle_col = df.columns[1]
setting_cols = df.columns[2:6]  # 前4列为工况参数
sensor_cols = df.columns[6:]

print(f"工况参数列: {setting_cols.tolist()}")
print(f"传感器列: {sensor_cols.tolist()}")
# === 2. 只处理 unit 16 的数据 ===
target_unit = 14
unit_data = df[df[unit_col] == target_unit]

if unit_data.empty:
    print(f"❌ 未找到 unit {target_unit} 的数据！")
    exit()

# print(f"✅ 找到 unit {target_unit} 的数据，共 {len(unit_data)} 行")

# === 3. 特征提取函数 ===
def calculate_sensor_features(cycle_data, sensor_cols):
    """计算传感器特征"""
    features = {}

    for sensor in sensor_cols:
        data = cycle_data[sensor].values

        # 基本统计特征
        features[f'{sensor}_mean'] = np.mean(data)
        features[f'{sensor}_std'] = np.std(data)
        features[f'{sensor}_max'] = np.max(data)
        features[f'{sensor}_min'] = np.min(data)
        features[f'{sensor}_ptp'] = np.ptp(data)  # 峰-峰值

        # 高阶统计特征
        features[f'{sensor}_skew'] = skew(data) # 偏度
        features[f'{sensor}_kurtosis'] = kurtosis(data) # 峰度

        # 首尾差值
        features[f'{sensor}_start_end_diff'] = data[-1] - data[0] if len(data) > 1 else 0

        # 计算斜率（线性拟合）
        if len(data) > 1:
            x = np.arange(len(data)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(x, data)
            features[f'{sensor}_slope'] = model.coef_[0]
        else:
            features[f'{sensor}_slope'] = 0

        # 一阶导数特征
        if len(data) > 1:
            derivative = np.diff(data) # 一阶导数
            features[f'{sensor}_derivative_mean'] = np.mean(derivative) # 一阶导数均值
            features[f'{sensor}_derivative_var'] = np.var(derivative)  # 一阶导数方差
        else:
            features[f'{sensor}_derivative_mean'] = 0
            features[f'{sensor}_derivative_var'] = 0

    return features

def calculate_setting_features(cycle_data, setting_cols):
    """计算工况参数特征"""
    features = {}

    # 假设工况参数顺序: Fc, hs, alt, Mach
    # 如果列名不同，请根据实际情况调整

    # Fc - 直接取均值
    features['Fc_mean'] = np.mean(cycle_data[setting_cols[0]])

    # hs - 取均值
    features['hs_mean'] = np.mean(cycle_data[setting_cols[1]])

    # alt - 最大值、均值、斜率
    alt_data = cycle_data[setting_cols[2]].values
    features['alt_max'] = np.max(alt_data)
    features['alt_mean'] = np.mean(alt_data)

    if len(alt_data) > 1:
        x = np.arange(len(alt_data)).reshape(-1, 1)
        model = LinearRegression()
        model.fit(x, alt_data)
        features['alt_slope'] = model.coef_[0]
    else:
        features['alt_slope'] = 0

    # Mach - 最大值、均值、波动幅度（标准差）
    mach_data = cycle_data[setting_cols[3]].values
    features['mach_max'] = np.max(mach_data)
    features['mach_mean'] = np.mean(mach_data)
    features['mach_std'] = np.std(mach_data)

    return features

# === 4. 提取所有特征 ===
all_features = []

for cycle_id, cycle_data in unit_data.groupby(cycle_col):
    print(f"处理 cycle {cycle_id}...")

    # 初始化特征字典
    cycle_features = {'cycle': cycle_id}

    # 计算传感器特征
    sensor_features = calculate_sensor_features(cycle_data, sensor_cols)
    cycle_features.update(sensor_features)

    # 计算工况参数特征
    setting_features = calculate_setting_features(cycle_data, setting_cols)
    cycle_features.update(setting_features)

    all_features.append(cycle_features)

# 转换为DataFrame
features_df = pd.DataFrame(all_features)

# 设置cycle为索引
features_df.set_index('cycle', inplace=True)

# === 5. 保存特征到CSV ===
csv_path = os.path.join(save_dir, f'{target_unit}_feature_all.csv')
features_df.to_csv(csv_path)
print(f"✅ 特征已保存到: {csv_path}")

# === 6. 绘制特征图 ===
print("开始绘制特征图...")

# 计算子图布局
n_features = len(features_df.columns)
n_cols = 4
n_rows = math.ceil(n_features / n_cols)

# 创建大图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

# 如果只有一行，确保axes是二维数组
if n_rows == 1:
    axes = axes.reshape(1, -1)

# 绘制每个特征
for i, feature in enumerate(features_df.columns):
    row = i // n_cols
    col = i % n_cols

    axes[row, col].plot(features_df.index, features_df[feature], linewidth=1.5, marker='o', markersize=3)
    axes[row, col].set_title(f'{feature}', fontsize=10, fontweight='bold')
    axes[row, col].set_xlabel('Cycle', fontsize=8)
    axes[row, col].set_ylabel('Value', fontsize=8)
    axes[row, col].grid(True, alpha=0.3)

    # 旋转x轴标签以避免重叠
    plt.setp(axes[row, col].xaxis.get_majorticklabels(), rotation=45)

# 隐藏多余的子图
for i in range(n_features, n_rows * n_cols):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].set_visible(False)

# 设置总标题
fig.suptitle(f'Unit {target_unit} - All Features', fontsize=16, fontweight='bold', y=0.98)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # 为总标题留出空间

# 保存特征图
plot_path = os.path.join(save_dir, f'{target_unit}_all_features.png')
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"✅ 特征图已保存到: {plot_path}")

# === 7. 按特征类型分别绘图 ===
print("开始按类型绘制特征图...")

# 按特征类型分组
sensor_mean_features = [col for col in features_df.columns if '_mean' in col and not col.startswith(('Fc', 'hs', 'alt', 'mach'))]
sensor_std_features = [col for col in features_df.columns if '_std' in col]
sensor_other_features = [col for col in features_df.columns if any(x in col for x in ['_max', '_min', '_ptp', '_skew', '_kurtosis', '_slope', '_diff', '_derivative'])]

setting_features = [col for col in features_df.columns if col.startswith(('Fc', 'hs', 'alt', 'mach'))]

# 绘制传感器均值特征
if sensor_mean_features:
    n_cols = 4
    n_rows = math.ceil(len(sensor_mean_features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(sensor_mean_features):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].plot(features_df.index, features_df[feature], linewidth=1.5, color='blue')
        axes[row, col].set_title(f'{feature}', fontsize=10)
        axes[row, col].grid(True, alpha=0.3)
        plt.setp(axes[row, col].xaxis.get_majorticklabels(), rotation=45)

    # 隐藏多余子图
    for i in range(len(sensor_mean_features), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Unit {target_unit} - Sensor Mean Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_dir, f'{target_unit}_sensor_mean_features.png'), dpi=200, bbox_inches='tight')
    plt.close()

# 绘制工况参数特征
if setting_features:
    n_cols = 3
    n_rows = math.ceil(len(setting_features) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, feature in enumerate(setting_features):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].plot(features_df.index, features_df[feature], linewidth=1.5, color='green')
        axes[row, col].set_title(f'{feature}', fontsize=10)
        axes[row, col].grid(True, alpha=0.3)
        plt.setp(axes[row, col].xaxis.get_majorticklabels(), rotation=45)

    # 隐藏多余子图
    for i in range(len(setting_features), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f'Unit {target_unit} - Setting Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(os.path.join(save_dir, f'{target_unit}_setting_features.png'), dpi=200, bbox_inches='tight')
    plt.close()

print("所有处理完成！")