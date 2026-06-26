import os
import numpy as np
import pandas as pd
import pickle


def prepare_simple_sequences(features_df, feature_cols, time_col='cycle', seq_len=30):
    """
    简化的序列构建函数 - 假设数据已经干净
    """
    df = features_df.copy()

    # 确保按时间排序
    df = df.sort_values(time_col).reset_index(drop=True)

    # 标准化特征
    for f in feature_cols:
        mu = df[f].mean()
        sigma = df[f].std() if df[f].std() > 0 else 1.0
        df[f + '_normalized'] = (df[f] - mu) / sigma

    # 构建序列
    sequences = []
    T = len(df)

    for end in range(seq_len, T):
        window = df.iloc[end - seq_len:end]
        sequence = []

        for _, row in window.iterrows():
            time_vector = []
            for f in feature_cols:
                time_vector.append(row[f + '_normalized'])
            sequence.append(time_vector)

        sequences.append(np.array(sequence, dtype=np.float32))

    X_seq = np.array(sequences)  # (num_samples, seq_len, num_features)

    print(f"序列数据形状: {X_seq.shape}")
    print(f"样本数: {X_seq.shape[0]}")
    print(f"序列长度: {X_seq.shape[1]}")
    print(f"特征维度: {X_seq.shape[2]}")

    return df, X_seq


# 使用示例
if __name__ == "__main__":
    # 读取数据
    df = pd.read_csv("feature_all/unit15/15_feature_all.csv")

    # 读取特征选择结果
    selected_features = [...]  # 从您的特征选择结果加载

    # 构建序列
    df_processed, sequences = prepare_simple_sequences(
        df,
        selected_features,
        time_col='cycle',
        seq_len=30
    )

    # 保存结果
    np.save("sequences.npy", sequences)
    df_processed.to_csv("processed_features.csv", index=False)