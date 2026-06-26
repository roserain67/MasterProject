import numpy as np
import pandas as pd
import pickle
import os


def quality_enhancement_pipeline(df, selected_features):
    df_enhanced = df.copy()

    # 确保数据按cycle排序
    if 'cycle' in df_enhanced.columns:
        df_enhanced = df_enhanced.sort_values('cycle').reset_index(drop=True)

    # 计算训练统计量（基于当前数据）
    train_stats = {}
    for f in selected_features:
        train_stats[f] = {
            'mean': df_enhanced[f].mean(),
            'std': df_enhanced[f].std() if df_enhanced[f].std() > 0 else 1.0
        }

    # 对每个选定特征进行质量增强
    for f in selected_features:
        print(f"处理特征: {f}")

        # 1) 把 inf -> NaN，但先记录 inf_flag
        df_enhanced[f + "_inf_flag"] = np.isinf(df_enhanced[f]).astype(int)
        df_enhanced[f] = df_enhanced[f].replace([np.inf, -np.inf], np.nan)

        # 2) 构造 mask
        df_enhanced[f + "_mask"] = (~df_enhanced[f].isna()).astype(int)

        # 3) delta: 在整个数据集中计算距离上次观测的时间步数
        def compute_delta_global(series):
            last = -9999
            deltas = []
            for i, val in enumerate(series.values):
                if not np.isnan(val):
                    last = i
                    deltas.append(0)
                else:
                    if last == -9999:
                        deltas.append(np.nan)  # never observed
                    else:
                        deltas.append(i - last)
            return deltas

        delta_values = compute_delta_global(df_enhanced[f])
        df_enhanced[f + "_delta"] = delta_values

        # 处理delta中的NaN（从未观测的情况）
        max_delta = df_enhanced[f + "_delta"].max()
        if pd.isna(max_delta) or max_delta == 0:
            max_delta = len(df_enhanced)  # 使用数据长度作为最大值
        df_enhanced[f + "_delta"] = df_enhanced[f + "_delta"].fillna(max_delta)

        # 4) impute: 前向填充，如果没有前值则用列中位数
        df_enhanced[f + "_imputed"] = df_enhanced[f].fillna(method='ffill').fillna(df_enhanced[f].median())

        # 5) outlier: 基于训练统计量（均值/标准差）
        z = (df_enhanced[f + "_imputed"] - train_stats[f]['mean']) / (train_stats[f]['std'] + 1e-12)
        df_enhanced[f + "_outlier"] = (np.abs(z) > 5).astype(int)

        print(f"  - 缺失值: {df_enhanced[f].isna().sum()}")
        print(f"  - 异常值: {df_enhanced[f + '_outlier'].sum()}")
        print(f"  - 无穷大标记: {df_enhanced[f + '_inf_flag'].sum()}")

    return df_enhanced


def build_feature_vectors(df_enhanced, selected_features):
    """
    构建最终的特征向量
    """
    feature_vectors = []

    for f in selected_features:
        imputed_vals = df_enhanced[f + "_imputed"].values.reshape(-1, 1)
        mask_vals = df_enhanced[f + "_mask"].values.reshape(-1, 1)
        delta_vals = df_enhanced[f + "_delta"].values.reshape(-1, 1)
        outlier_vals = df_enhanced[f + "_outlier"].values.reshape(-1, 1)

        # 组合单个特征的4个分量
        feature_vector = np.concatenate([imputed_vals, mask_vals, delta_vals, outlier_vals], axis=1)
        feature_vectors.append(feature_vector)

    # 组合所有特征
    final_vectors = np.concatenate(feature_vectors, axis=1)

    print(f"特征向量形状: {final_vectors.shape}")
    print(f"时间步数 (T): {final_vectors.shape[0]}")
    print(f"特征维度 (D): {final_vectors.shape[1]}")

    return final_vectors


def create_sequences(feature_vectors, seq_len=30):
    """
    创建时间序列样本
    """
    sequences = []
    T = len(feature_vectors)

    for i in range(seq_len, T):
        sequence = feature_vectors[i - seq_len:i]
        sequences.append(sequence)

    X_seq = np.array(sequences)
    print(f"序列数据形状: {X_seq.shape}")  # (样本数, 序列长度, 特征数)

    return X_seq


# ================
# 主程序
# ================
if __name__ == "__main__":
    # 参数设置
    data_dir = "feature_all/unit14"
    feature_selected_dir = os.path.join(data_dir, "feature_selected")

    # 1. 读取原始数据
    original_data_path = os.path.join(data_dir, "14_feature_all_freq.csv")
    df_original = pd.read_csv(original_data_path)

    # 如果cycle是索引，重置为列
    if df_original.index.name == 'cycle':
        df_original = df_original.reset_index()

    print(f"原始数据形状: {df_original.shape}")

    # 2. 读取特征选择结果
    selected_features_path = os.path.join(feature_selected_dir, "selected_features.pkl")
    with open(selected_features_path, "rb") as f:
        selected_features = pickle.load(f)

    print(f"选定的特征数量: {len(selected_features)}")
    print(f"选定的特征: {selected_features}")

    # 3. 应用质量增强流水线
    print("\n=== 开始质量增强 ===")
    df_enhanced = quality_enhancement_pipeline(df_original, selected_features)

    # 4. 构建特征向量
    print("\n=== 构建特征向量 ===")
    feature_vectors = build_feature_vectors(df_enhanced, selected_features)

    # 5. 创建时间序列
    print("\n=== 创建时间序列 ===")
    sequences = create_sequences(feature_vectors, seq_len=30)

    # 6. 保存结果
    print("\n=== 保存结果 ===")

    # 保存增强后的DataFrame
    enhanced_data_path = os.path.join(feature_selected_dir, "enhanced_features.csv")
    df_enhanced.to_csv(enhanced_data_path, index=False)
    print(f"✓ 增强数据已保存: {enhanced_data_path}")

    # 保存序列数据
    sequences_path = os.path.join(feature_selected_dir, "sequences.npy")
    np.save(sequences_path, sequences)
    print(f"✓ 序列数据已保存: {sequences_path}")

    # 保存特征向量
    vectors_path = os.path.join(feature_selected_dir, "feature_vectors.npy")
    np.save(vectors_path, feature_vectors)
    print(f"✓ 特征向量已保存: {vectors_path}")

    print("\n=== 处理完成 ===")
    print(f"最终序列形状: {sequences.shape}")
    print(f"样本数: {sequences.shape[0]}")
    print(f"序列长度: {sequences.shape[1]}")
    print(f"特征维度: {sequences.shape[2]}")
    print(f"可用于GRU/LSTM训练")