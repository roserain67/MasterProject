import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.linear_model import LinearRegression

# === 参数区 ===
file_path = "D:\\yyo-Python\\0毕设\\1数据处理\\DS02\\N-CMAPSS_DS02_selected.csv"
save_dir = "feature_all\\unit15"
os.makedirs(save_dir, exist_ok=True)

# === 1. 读取数据 ===
df = pd.read_csv(file_path)

unit_col = df.columns[0]
cycle_col = df.columns[1]

setting_cols = df.columns[2:6]   # 工况参数
sensor_cols = df.columns[6:]     # 传感器

target_unit = 15
unit_data = df[df[unit_col] == target_unit]

if unit_data.empty:
    print(f"❌ 未找到 unit {target_unit} 的数据")
    exit()

# === 3. 特征提取函数（核心） ===
def compute_welch_features(data, fs=1.0):
    """
    Welch PSD：peak_freq, centroid, entropy, energies, bandwidth
    若数据 < 32点，则放弃频域特征
    """
    features = {}
    if len(data) < 32:
        # 数据太短不做频域提取
        features.update({
            "peak_freq": np.nan,
            "spectral_centroid": np.nan,
            "spectral_entropy": np.nan,
            "band1_energy": np.nan,
            "band2_energy": np.nan,
            "band3_energy": np.nan,
            "bandwidth": np.nan
        })
        return features

    freqs, psd = welch(data, fs=fs, nperseg=min(256, len(data)))

    # 1. 最大能量频率
    peak_freq = freqs[np.argmax(psd)]

    # 2. 频谱质心
    spectral_centroid = np.sum(freqs * psd) / np.sum(psd)

    # 3. 频谱熵
    p = psd / np.sum(psd)
    spectral_entropy = -np.sum(p * np.log(p + 1e-12))

    # 4. 分段能量（0–1Hz, 1–2Hz, >2Hz）
    band1_energy = np.sum(psd[(freqs >= 0) & (freqs < 1)])
    band2_energy = np.sum(psd[(freqs >= 1) & (freqs < 2)])
    band3_energy = np.sum(psd[(freqs >= 2)])

    # 5. 带宽
    bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))

    features.update({
        "peak_freq": peak_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy,
        "band1_energy": band1_energy,
        "band2_energy": band2_energy,
        "band3_energy": band3_energy,
        "bandwidth": bandwidth
    })
    return features


def calculate_sensor_features(cycle_data, sensor_cols):
    """计算传感器高级特征（包含所有你要求的内容）"""
    features = {}

    for sensor in sensor_cols:
        x = cycle_data[sensor].values
        L = len(x)

        features[f"{sensor}_mean"] = np.mean(x)
        features[f"{sensor}_std"] = np.std(x)
        features[f"{sensor}_median"] = np.median(x)

        # IQR
        q25, q75 = np.percentile(x, [25, 75])
        features[f"{sensor}_iqr"] = q75 - q25

        # 峰峰值
        features[f"{sensor}_ptp"] = np.ptp(x)

        # 均方根
        features[f"{sensor}_rms"] = np.sqrt(np.mean(x**2))

        # 偏度/峰度
        features[f"{sensor}_skew"] = skew(x)
        features[f"{sensor}_kurtosis"] = kurtosis(x)

        # peak count（局部极值）
        peaks = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0]
        features[f"{sensor}_peak_count"] = len(peaks)

        # slope + R²
        if L > 1:
            t = np.arange(L).reshape(-1, 1)
            model = LinearRegression().fit(t, x)
            slope = model.coef_[0]
            pred = model.predict(t)
            ss_res = np.sum((x - pred)**2)
            ss_tot = np.sum((x - np.mean(x))**2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
        else:
            slope, r2 = 0, 0

        features[f"{sensor}_slope"] = slope
        features[f"{sensor}_r2"] = r2

        # 一阶差分
        if L > 1:
            d = np.diff(x)
            features[f"{sensor}_diff_mean"] = np.mean(d)
            features[f"{sensor}_diff_std"] = np.std(d)
        else:
            features[f"{sensor}_diff_mean"] = 0
            features[f"{sensor}_diff_std"] = 0

        # 频域 Welch PSD 特征
        psd_feats = compute_welch_features(x, fs=1.0)
        for k, v in psd_feats.items():
            features[f"{sensor}_{k}"] = v

    return features


def calculate_setting_features(cycle_data, setting_cols):
    """工况特征：保证保留工况信息用于 PEARL 的任务推断"""
    feats = {}
    for col in setting_cols:
        x = cycle_data[col].values
        feats[f"{col}_mean"] = np.mean(x)
        feats[f"{col}_std"] = np.std(x)
        feats[f"{col}_min"] = np.min(x)
        feats[f"{col}_max"] = np.max(x)
    return feats


# ============ 4. 提取全部特征 ============
all_features = []

for cycle_id, cycle_data in unit_data.groupby(cycle_col):
    f = {"cycle": cycle_id}

    f.update(calculate_sensor_features(cycle_data, sensor_cols))
    f.update(calculate_setting_features(cycle_data, setting_cols))

    all_features.append(f)

features_df = pd.DataFrame(all_features).set_index("cycle")

# === 保存 CSV ===
csv_path = os.path.join(save_dir, f"{target_unit}_feature_all_freq.csv")
features_df.to_csv(csv_path)
print(f"特征保存至: {csv_path}")
