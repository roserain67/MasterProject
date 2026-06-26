# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import mutual_info_regression
# from sklearn.preprocessing import StandardScaler
# import pickle
# import os
#
# # === 参数 ===
# feature_csv = "feature_all/unit14/14_feature_all.csv"
# save_dir = "feature_all/unit14/feature_selected"
# os.makedirs(save_dir, exist_ok=True)
#
# K = 32  # 最终保留的特征数量
#
# # 工况（task）特征，必须保留
# mandatory_setting_features = [
#     "alt_mean", "alt_std",
#     "Mach_mean", "Mach_std",
#     "TRA_mean", "TRA_std",
#     "T2_mean", "T2_std"
# ]
#
# # 健康参数（HPT LPT fan）
# health_features = ["HPT_eff_mod_mean", "LPT_flow_mod_mean", "LPT_eff_mod_mean"]
#
# # ================
# # Step 1. 读取特征
# # ================
# df = pd.read_csv(feature_csv, index_col="cycle")
# print(f"原始特征数量: {len(df.columns)}")
# print(f"原始数据列名: {list(df.columns)}")
#
# # 检查健康参数是否存在
# print(f"健康参数检查:")
# for hp in health_features:
#     if hp in df.columns:
#         print(f"  ✓ {hp} 存在")
#     else:
#         print(f"  ✗ {hp} 不存在")
#
# # ===========================
# # Step 2. 弱变化特征删除（排除强制保留特征和健康参数）
# # ===========================
# # 只对非强制、非健康参数的特征进行过滤
# features_to_filter = [col for col in df.columns
#                       if col not in mandatory_setting_features + health_features]
#
# var = df[features_to_filter].var()
# low_var_cols = var[var < 1e-6].index.tolist()
# df_filtered = df.drop(columns=low_var_cols)
#
# print(f"删除低方差特征 {len(low_var_cols)} 个，剩余特征: {len(df_filtered.columns)}")
#
# # ===========================
# # Step 3. 删除强相关特征（只对非强制、非健康参数的特征）
# # ===========================
# # 确保健康参数和强制特征不会被删除
# protected_features = mandatory_setting_features + health_features
# filterable_features = [col for col in df_filtered.columns
#                        if col not in protected_features]
#
# print(f"可过滤特征数量: {len(filterable_features)}")
# print(f"受保护特征数量: {len(protected_features)}")
#
# if filterable_features:
#     corr = df_filtered[filterable_features].corr().abs()
#     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#
#     high_corr_cols = []
#     for col in upper.columns:
#         if any(upper[col] > 0.90):
#             high_corr_cols.append(col)
#
#     df_reduced = df_filtered.drop(columns=high_corr_cols)
#     print(f"删除高相关特征 {len(high_corr_cols)} 个，剩余特征: {len(df_reduced.columns)}")
# else:
#     df_reduced = df_filtered
#     print("没有可过滤的特征，跳过相关性删除")
#
# # 再次检查健康参数是否存在
# print(f"过滤后健康参数检查:")
# available_health_features = []
# for hp in health_features:
#     if hp in df_reduced.columns:
#         available_health_features.append(hp)
#         print(f"  ✓ {hp} 存在")
#     else:
#         print(f"  ✗ {hp} 不存在 - 可能被误删除")
#
# # ===========================
# # Step 4. MI 互信息评分
# # ===========================
# if available_health_features:
#     # 准备特征数据（排除健康参数）
#     features_for_mi = df_reduced.drop(columns=available_health_features, errors='ignore')
#
#     print(f"用于MI计算的特征数量: {len(features_for_mi.columns)}")
#
#     # 标准化
#     scaler = StandardScaler()
#     X = scaler.fit_transform(features_for_mi)
#
#     # 多目标互信息计算
#     mi_scores_combined = np.zeros(len(features_for_mi.columns))
#
#     for health_target in available_health_features:
#         target_values = df_reduced[health_target]
#         mi_scores = mutual_info_regression(X, target_values, random_state=42)
#         mi_scores_combined += mi_scores
#
#     # 平均互信息分数
#     mi_scores_combined /= len(available_health_features)
#
#     mi_series = pd.Series(mi_scores_combined, index=features_for_mi.columns)
#     mi_series = mi_series.sort_values(ascending=False)
#
#     print(f"基于 {len(available_health_features)} 个健康参数计算平均互信息")
#
#     # 选择前 K 个特征
#     topK = mi_series.head(K).index.tolist()
#     print(f"MI选择的前{K}个特征: {topK}")
#
# else:
#     print("⚠ 没有可用的健康参数，使用方差进行特征选择")
#     # 备用方案：使用方差选择
#     features_for_fallback = [col for col in df_reduced.columns
#                              if col not in mandatory_setting_features]
#     var_scores = df_reduced[features_for_fallback].var().sort_values(ascending=False)
#     topK = var_scores.head(K).index.tolist()
#     print(f"方差选择的前{K}个特征: {topK}")
#
# # ================
# # Step 5. 强制保留关键特征
# # ================
# # 确保强制特征存在
# existing_mandatory = [f for f in mandatory_setting_features if f in df_reduced.columns]
# existing_health = [f for f in health_features if f in df_reduced.columns]
#
# selected_features = list(set(topK + existing_mandatory + existing_health))
#
# print(f"\n=== 特征选择结果 ===")
# print(f"强制保留工况特征: {len(existing_mandatory)}/{len(mandatory_setting_features)}")
# print(f"健康参数: {len(existing_health)}/{len(health_features)}")
# print(f"互信息选择特征: {len(topK)}")
# print(f"最终特征总数: {len(selected_features)}")
#
# # 检查是否有特征在最终选择中不存在于原始数据
# missing_in_final = [f for f in selected_features if f not in df_reduced.columns]
# if missing_in_final:
#     print(f"⚠ 警告: 以下特征在最终数据中不存在，将被移除: {missing_in_final}")
#     selected_features = [f for f in selected_features if f not in missing_in_final]
#
# # 保存选择列表
# with open(os.path.join(save_dir, "selected_features.pkl"), "wb") as f:
#     pickle.dump(selected_features, f)
#
# # 保存特征重要性分析
# if 'mi_series' in locals():
#     feature_importance_df = pd.DataFrame({
#         'feature': mi_series.index,
#         'mi_score': mi_series.values
#     })
#     feature_importance_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
#     print(f"✓ 特征重要性已保存: feature_importance.csv")
#
# print("✓ 自动特征选择完成")
# print("✓ 已保存: selected_features.pkl")
# print(f"最终选择的特征: {selected_features}")

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import pickle
import os

# === 参数 ===
feature_csv = "D:\yyo-Python/0毕设/1数据处理/DS02/feature_all/unit15/15_feature_all.csv"
save_dir = "feature_all/unit15/feature_selected"
os.makedirs(save_dir, exist_ok=True)

K = 32

# 强制保留的特征
mandatory_setting_features = [
    "alt_mean", "alt_std", "Mach_mean", "Mach_std",
    "TRA_mean", "TRA_std", "T2_mean", "T2_std"
]

health_features = ["HPT_eff_mod_mean", "LPT_flow_mod_mean", "LPT_eff_mod_mean"]

# ================
# Step 1. 读取特征
# ================
df = pd.read_csv(feature_csv, index_col="cycle")
print(f"原始特征数量: {len(df.columns)}")
print(f"原始数据行数: {len(df)}")

# 检查健康参数
available_health = [hp for hp in health_features if hp in df.columns]
print(f"可用的健康参数: {available_health}")

# ===========================
# Step 2. 数据清洗 - 处理NaN和无穷大值
# ===========================
print("\n=== 数据清洗 ===")

# 检查NaN值
nan_percentage = df.isna().sum() / len(df) * 100
cols_with_100_nan = nan_percentage[nan_percentage == 100].index.tolist()
cols_with_partial_nan = nan_percentage[(nan_percentage > 0) & (nan_percentage < 100)].index.tolist()

print(f"100% NaN的特征数量: {len(cols_with_100_nan)}")
print(f"部分NaN的特征数量: {len(cols_with_partial_nan)}")

# 检查无穷大值
inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
cols_with_inf = inf_counts[inf_counts > 0].index.tolist()
print(f"包含无穷大值的特征数量: {len(cols_with_inf)}")


# 数据清洗策略
def clean_data(df, protected_features):
    """清洗数据：删除100% NaN的列，对部分NaN的列进行填充"""
    df_clean = df.copy()

    # 1. 处理无穷大值
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # 2. 删除100% NaN的列
    nan_percentage = df_clean.isna().sum() / len(df_clean) * 100
    cols_to_drop = nan_percentage[nan_percentage == 100].index.tolist()

    # 保护重要特征不被删除，即使有100% NaN
    cols_to_drop = [col for col in cols_to_drop if col not in protected_features]

    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"删除100% NaN的特征: {len(cols_to_drop)}个")

    # 3. 对部分NaN的列进行填充
    # 首先处理保护特征
    for col in protected_features:
        if col in df_clean.columns and df_clean[col].isna().any():
            # 对于保护特征，用均值填充
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
            print(f"填充保护特征 {col} 的NaN值")

    # 然后处理其他特征
    other_cols = [col for col in df_clean.columns if col not in protected_features]
    for col in other_cols:
        if df_clean[col].isna().any():
            # 对于其他特征，用均值填充
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    # 4. 验证清洗结果
    remaining_nan = df_clean.isna().sum().sum()
    if remaining_nan > 0:
        print(f"警告: 清洗后仍有 {remaining_nan} 个NaN值")
    else:
        print("✓ 所有NaN值已处理")

    print(f"清洗后特征数量: {len(df_clean.columns)}")
    print(f"清洗后数据行数: {len(df_clean)}")

    return df_clean


# 保护重要特征不被删除
protected_features = mandatory_setting_features + available_health
df_clean = clean_data(df, protected_features)

# ===========================
# Step 3. 弱变化特征删除
# ===========================
print("\n=== 特征过滤 ===")
features_to_filter = [col for col in df_clean.columns if col not in protected_features]

var_threshold = 1e-6
variances = df_clean[features_to_filter].var()
low_var_features = variances[variances < var_threshold].index.tolist()

df_filtered = df_clean.drop(columns=low_var_features)
print(f"删除低方差特征: {len(low_var_features)}个")

# ===========================
# Step 4. 相关性过滤
# ===========================
filterable_features = [col for col in df_filtered.columns if col not in protected_features]

if filterable_features:
    # 计算相关性矩阵
    corr_matrix = df_filtered[filterable_features].corr().abs()

    # 找出高相关特征
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                colname = corr_matrix.columns[j]
                to_drop.add(colname)

    df_final = df_filtered.drop(columns=list(to_drop))
    print(f"删除高相关特征: {len(to_drop)}个")
else:
    df_final = df_filtered

print(f"最终数据特征数: {len(df_final.columns)}")
print(f"最终数据行数: {len(df_final)}")

# ===========================
# Step 5. MI特征选择
# ===========================
print("\n=== 特征选择 ===")

if available_health and len(df_final) > 10:  # 确保有足够的数据
    # 准备数据
    features_for_selection = [col for col in df_final.columns
                              if col not in available_health and col in df_final.columns]

    X = df_final[features_for_selection]
    y = df_final[available_health[0]]  # 使用第一个健康参数

    # 再次检查数据质量
    print(f"MI计算数据检查:")
    print(f"  X形状: {X.shape}, NaN数量: {X.isna().sum().sum()}")
    print(f"  y形状: {y.shape}, NaN数量: {y.isna().sum()}")

    if X.isna().sum().sum() > 0 or y.isna().sum() > 0:
        print("⚠ 数据仍然包含NaN，使用备用方案")
        # 备用方案：使用方差选择
        var_scores = X.var().sort_values(ascending=False)
        top_features = var_scores.head(K).index.tolist()
    else:
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 检查标准化后的数据
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            print("⚠ 标准化后出现NaN/Inf，使用备用方案")
            var_scores = X.var().sort_values(ascending=False)
            top_features = var_scores.head(K).index.tolist()
        else:
            # 计算互信息
            try:
                mi_scores = mutual_info_regression(X_scaled, y, random_state=42)
                mi_series = pd.Series(mi_scores, index=features_for_selection)
                mi_series = mi_series.sort_values(ascending=False)

                top_features = mi_series.head(K).index.tolist()
                print(f"✓ MI特征选择完成，最高MI分数: {mi_series.max():.4f}")

                # 保存特征重要性
                feature_importance_df = pd.DataFrame({
                    'feature': mi_series.index,
                    'mi_score': mi_series.values
                })
                feature_importance_df.to_csv(os.path.join(save_dir, "feature_importance.csv"), index=False)
                print(f"✓ 特征重要性已保存: feature_importance.csv")

            except Exception as e:
                print(f"⚠ MI计算失败: {e}，使用方差选择")
                var_scores = X.var().sort_values(ascending=False)
                top_features = var_scores.head(K).index.tolist()
else:
    print("⚠ 健康参数不可用或数据不足，使用方差选择")
    features_for_selection = [col for col in df_final.columns if col not in protected_features]
    var_scores = df_final[features_for_selection].var().sort_values(ascending=False)
    top_features = var_scores.head(K).index.tolist()

# ===========================
# Step 6. 组合最终特征
# ===========================
# 确保保护特征存在
existing_protected = [f for f in protected_features if f in df_final.columns]
selected_features = list(set(top_features + existing_protected))

print(f"\n=== 最终结果 ===")
print(f"选择特征总数: {len(selected_features)}")
print(f"MI选择特征: {len(top_features)}")
print(f"保护特征: {len(existing_protected)}")

# 分类统计
setting_features = [f for f in selected_features if f in mandatory_setting_features]
health_features_selected = [f for f in selected_features if f in health_features]
sensor_features = [f for f in selected_features if f not in setting_features + health_features_selected]

print(f"\n=== 特征分类 ===")
print(f"工况特征: {len(setting_features)}")
print(f"健康参数: {len(health_features_selected)}")
print(f"传感器特征: {len(sensor_features)}")

# 保存结果
with open(os.path.join(save_dir, "selected_features.pkl"), "wb") as f:
    pickle.dump(selected_features, f)

# 保存清洗后的数据（可选）
df_final[selected_features].to_csv(os.path.join(save_dir, "cleaned_features.csv"))

print(f"\n✓ 特征选择完成!")
print(f"✓ 已保存: selected_features.pkl")
print(f"✓ 已保存: cleaned_features.csv")

# 输出特征选择过程的总结
print(f"\n=== 处理总结 ===")
print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
print(f"清洗后数据: {len(df_final)} 行, {len(df_final.columns)} 列")
print(f"最终选择: {len(selected_features)} 个特征")