import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# === 参数 ===
feature_csv = "feature_all/unit15/15_feature_all_freq.csv"
save_dir = "feature_all/unit15/feature_selected"
os.makedirs(save_dir, exist_ok=True)

K = 32  # 最终保留的传感器特征数量

# 工况特征 - 必须保留
mandatory_setting_features = [
    "alt_mean", "alt_std",
    "Mach_mean", "Mach_std",
    "TRA_mean", "TRA_std",
    "T2_mean", "T2_std"
]

# 健康参数 - 必须保留
health_features = ["HPT_eff_mod_mean", "LPT_flow_mod_mean", "LPT_eff_mod_mean"]

# ================
# Step 1. 读取数据
# ================
df = pd.read_csv(feature_csv, index_col="cycle")
print(f"原始特征数量: {len(df.columns)}")
print(f"原始数据行数: {len(df)}")

# 检查强制特征是否存在
print("\n=== 强制特征检查 ===")
existing_mandatory = [f for f in mandatory_setting_features if f in df.columns]
existing_health = [f for f in health_features if f in df.columns]

print(f"工况特征: {len(existing_mandatory)}/{len(mandatory_setting_features)}")
print(f"健康参数: {len(existing_health)}/{len(health_features)}")

# ===========================
# Step 2. 删除100% NaN的特征列
# ===========================
print("\n=== 删除100% NaN特征 ===")
nan_percentage = df.isna().sum() / len(df) * 100
cols_with_100_nan = nan_percentage[nan_percentage == 100].index.tolist()

# 保护强制特征不被删除
cols_to_drop = [col for col in cols_with_100_nan
                if col not in existing_mandatory + existing_health]

if cols_to_drop:
    df_clean = df.drop(columns=cols_to_drop)
    print(f"删除100% NaN特征: {len(cols_to_drop)}个")
    print(f"删除的特征: {cols_to_drop}")
else:
    df_clean = df.copy()
    print("没有发现100% NaN的特征")

print(f"清理后特征数量: {len(df_clean.columns)}")

# Step 3. 识别传感器特征
# ===========================
# 所有特征减去强制特征
all_features = df_clean.columns.tolist()
sensor_features = [f for f in all_features
                   if f not in existing_mandatory + existing_health]

print(f"\n=== 特征分类 ===")
print(f"工况特征: {len(existing_mandatory)}")
print(f"健康参数: {len(existing_health)}")
print(f"传感器特征: {len(sensor_features)}")

# Step 4. 删除传感器特征中的强相关特征
# ===========================
print("\n=== 相关性过滤 ===")
if len(sensor_features) > 1:
    # 计算传感器特征的相关性矩阵
    corr_matrix = df_clean[sensor_features].corr().abs()

    # 找出高相关特征对 (阈值0.95)
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                colname = corr_matrix.columns[j]
                # 保留方差较大的特征，删除方差较小的
                if df_clean[corr_matrix.columns[i]].var() > df_clean[colname].var():
                    to_drop.add(colname)
                else:
                    to_drop.add(corr_matrix.columns[i])

    if to_drop:
        sensor_features_reduced = [f for f in sensor_features if f not in to_drop]
        print(f"删除高相关传感器特征: {len(to_drop)}个")
        print(f"删除的特征: {list(to_drop)}")
        print(f"相关性过滤后传感器特征: {len(sensor_features_reduced)}个")
    else:
        sensor_features_reduced = sensor_features
        print("没有发现高相关特征对")
else:
    sensor_features_reduced = sensor_features
    print("传感器特征数量不足，跳过相关性过滤")

# Step 5. MI互信息评分 - 强烈建议保留
# ===========================
print("\n=== MI互信息特征选择 ===")

if existing_health and len(sensor_features_reduced) > 0:
    # 准备数据
    X = df_clean[sensor_features_reduced]

    # 处理剩余NaN值 - 用中位数填充
    X_filled = X.fillna(X.median())

    # 检查无穷大值
    if np.isinf(X_filled).any().any():
        print("⚠ 数据包含无穷大值，使用备用方案")
        # 备用方案：使用方差选择
        var_scores = X_filled.var().sort_values(ascending=False)
        selected_sensor_features = var_scores.head(K).index.tolist()
    else:
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filled)

        # 多目标互信息计算
        mi_scores_combined = np.zeros(len(sensor_features_reduced))

        for health_target in existing_health:
            y = df_clean[health_target]
            # 填充目标变量的NaN值
            y_filled = y.fillna(y.median())

            try:
                mi_scores = mutual_info_regression(X_scaled, y_filled, random_state=42)
                mi_scores_combined += mi_scores
            except Exception as e:
                print(f"⚠ 计算{health_target}的MI时出错: {e}")
                continue

        if np.sum(mi_scores_combined) > 0:
            # 平均互信息分数
            mi_scores_combined /= len(existing_health)

            mi_series = pd.Series(mi_scores_combined, index=sensor_features_reduced)
            mi_series = mi_series.sort_values(ascending=False)

            # 选择前K个特征
            selected_sensor_features = mi_series.head(K).index.tolist()

            print(f"✓ MI特征选择完成，基于 {len(existing_health)} 个健康参数")
            print(f"最高MI分数: {mi_series.max():.4f}")
            print(f"最低MI分数: {mi_series.min():.4f}")

            # 保存特征重要性
            feature_importance_df = pd.DataFrame({
                'feature': mi_series.index,
                'mi_score': mi_series.values
            })
            feature_importance_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
            print(f"✓ 特征重要性已保存: feature_importance.csv")
        else:
            print("⚠ MI计算失败，使用方差选择")
            var_scores = X_filled.var().sort_values(ascending=False)
            selected_sensor_features = var_scores.head(K).index.tolist()
else:
    print("⚠ 健康参数不可用，使用方差选择")
    X_filled = df_clean[sensor_features_reduced].fillna(df_clean[sensor_features_reduced].median())
    var_scores = X_filled.var().sort_values(ascending=False)
    selected_sensor_features = var_scores.head(K).index.tolist()

print(f"选择的传感器特征数量: {len(selected_sensor_features)}")

# Step 6. 组合最终特征
selected_features = existing_mandatory + existing_health + selected_sensor_features

print(f"\n=== 最终特征选择结果 ===")
print(f"工况特征: {len(existing_mandatory)}")
print(f"健康参数: {len(existing_health)}")
print(f"传感器特征: {len(selected_sensor_features)}")
print(f"特征总数: {len(selected_features)}")

# 详细分类统计
sensor_feature_types = {}
for feature in selected_sensor_features:
    # 根据特征名分类
    if 'Nf_' in feature:
        sensor_feature_types.setdefault('风扇转速 (Nf)', []).append(feature)
    elif 'Nc_' in feature:
        sensor_feature_types.setdefault('核心机转速 (Nc)', []).append(feature)
    elif 'T24_' in feature:
        sensor_feature_types.setdefault('LPC出口温度 (T24)', []).append(feature)
    elif 'T48_' in feature:
        sensor_feature_types.setdefault('LPT出口温度 (T48)', []).append(feature)
    elif 'P30_' in feature:
        sensor_feature_types.setdefault('HPC出口压力 (P30)', []).append(feature)
    elif 'Wf_' in feature:
        sensor_feature_types.setdefault('燃油流量 (Wf)', []).append(feature)
    else:
        sensor_feature_types.setdefault('其他传感器', []).append(feature)

print("\n=== 传感器特征详细分类 ===")
for sensor_type, features in sensor_feature_types.items():
    print(f"{sensor_type}: {len(features)}个 -> {features}")

# 索引映射（按你确认的最终版本）
sensor_name_to_index = {
    'Nf_': 0,    # 风扇 (LPT 前端)
    'Nc_': 1,    # 核心机 (HPT 相关)
    'T24_': 2,   # LPC 出口温度 → LPT
    'T48_': 3,   # LPT 出口温度 → HPT/LPT 公共
    'P30_': 4,   # HPC 出口压力 → HPT
    'Wf_': 5     # 燃油流量 → HPT
}

print("\n=== 传感器特征索引 ===")
for sensor_type, features in sensor_feature_types.items():
    indices = set()
    for feature in features:
        for sensor_prefix, index in sensor_name_to_index.items():
            if sensor_prefix in feature:
                indices.add(index)
    print(f"{sensor_type}: 索引 {sorted(list(indices))}")

# 最终确认的 HPT / LPT 组件索引
idx_HPT = [sensor_name_to_index['Nc_'],
           sensor_name_to_index['T48_'],
           sensor_name_to_index['P30_'],
           sensor_name_to_index['Wf_']]

idx_LPT = [sensor_name_to_index['Nf_'],
           sensor_name_to_index['T24_'],
           sensor_name_to_index['T48_']]  # T48 双重用于 HPT/LPT

print("\n=== 最终 HPT & LPT 对应索引 ===")
print("HPT 传感器索引:", idx_HPT)
print("LPT 传感器索引:", idx_LPT)

# Step 7. 保存结果
# 保存选择列表
with open(os.path.join(save_dir, "selected_features.pkl"), "wb") as f:
    pickle.dump(selected_features, f)

# 保存处理后的数据（不填充NaN，留给后续质量增强步骤）
df_clean[selected_features].to_csv(os.path.join(save_dir, "selected_features_data.csv"))

print(f"\n✓ 特征选择完成!")
print(f"✓ 已保存: selected_features.pkl")
print(f"✓ 已保存: selected_features_data.csv")

print(f"\n=== 处理总结 ===")
print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
print(f"最终选择: {len(selected_features)} 个特征")
print(f"删除的100% NaN特征: {len(cols_to_drop)}个")
print(f"删除的高相关特征: {len(sensor_features) - len(sensor_features_reduced)}个")