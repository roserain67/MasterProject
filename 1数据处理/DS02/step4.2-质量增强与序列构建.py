import numpy as np
import pandas as pd
import pickle
import os
'''
端到端处理数据
'''
def prepare_for_gru_complete(features_df, feature_cols, time_col='cycle', seq_len=30, max_delta=30):
    """
    完整的数据预处理流水线 - 针对单个发动机单元
    """
    df = features_df.copy()

    # 确保数据按时间排序
    df = df.sort_values(time_col).reset_index(drop=True)

    # 1. 无穷大值处理与标记
    print("=== 处理无穷大值 ===")
    for f in feature_cols:
        df[f + '_inf_flag'] = np.isinf(df[f]).astype(int)
        inf_count = df[f + '_inf_flag'].sum()
        if inf_count > 0:
            print(f"特征 {f}: 发现 {inf_count} 个无穷大值")
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)

    # 2. 计算全局统计量
    print("=== 计算全局统计量 ===")
    global_median = df[feature_cols].median()
    global_mean = df[feature_cols].mean()
    global_std = df[feature_cols].std().replace(0, 1.0)

    # 3. 逐时间点处理数据质量
    print("=== 处理缺失值和异常值 ===")
    T = len(df)
    last_obs = {f: -9999 for f in feature_cols}
    processed_rows = []

    for t in range(T):
        row = df.iloc[t].copy()
        for f in feature_cols:
            val = df.at[t, f]

            # 处理缺失值
            if not pd.isna(val):
                row[f + '_mask'] = 1
                row[f + '_imputed'] = val
                last_obs[f] = t
            else:
                row[f + '_mask'] = 0
                if last_obs[f] == -9999:
                    row[f + '_imputed'] = global_median[f]  # 从未观测到
                    row[f + '_delta'] = max_delta
                else:
                    row[f + '_imputed'] = df.at[last_obs[f], f]  # 前向填充
                    row[f + '_delta'] = min(t - last_obs[f], max_delta)

            # 确保delta存在
            if f + '_delta' not in row:
                row[f + '_delta'] = 0 if row[f + '_mask'] == 1 else max_delta

            # 异常值检测
            z = (row[f + '_imputed'] - global_mean[f]) / (global_std[f] + 1e-12)
            row[f + '_outlier'] = 1 if abs(z) > 5 else 0

        processed_rows.append(row)

    df_prepared = pd.DataFrame(processed_rows)

    # 4. 数据标准化
    print("=== 数据标准化 ===")
    for f in feature_cols:
        mu = df_prepared[f + '_imputed'].mean()
        sigma = df_prepared[f + '_imputed'].std() if df_prepared[f + '_imputed'].std() > 0 else 1.0
        df_prepared[f + '_imputed'] = (df_prepared[f + '_imputed'] - mu) / sigma
        # 归一化delta
        df_prepared[f + '_delta'] = df_prepared[f + '_delta'] / max_delta

    # 5. 构建序列
    print("=== 构建时间序列 ===")
    sequences = []
    T = len(df_prepared)

    for end in range(seq_len - 1, T):
        window = df_prepared.iloc[end - seq_len + 1:end + 1]
        time_vectors = []

        for _, row in window.iterrows():
            v = []
            for f in feature_cols:
                v.append(row[f + '_imputed'])  # 标准化数值
                v.append(row[f + '_mask'])  # 缺失标志
                v.append(row[f + '_delta'])  # 缺失时长
                v.append(row[f + '_outlier'])  # 异常标志
            time_vectors.append(v)

        sequences.append(np.array(time_vectors, dtype=np.float32))

    if len(sequences) == 0:
        raise ValueError(f"无法构建序列，序列长度{seq_len}大于数据长度{T}")

    X_seq = np.stack(sequences, axis=0)  # (num_samples, seq_len, input_dim)

    return df_prepared, X_seq


def build_full_trajectory(df_prepared, feature_cols):
    """
    构建完整退化轨迹 (T, 172)，用于维护决策环境。
    pointer 0 = 寿命初期(健康)，pointer T-1 = 寿命末期(退化)。
    """
    T = len(df_prepared)
    trajectory = []
    for t in range(T):
        row = df_prepared.iloc[t]
        v = []
        for f in feature_cols:
            v.append(row[f + '_imputed'])
            v.append(row[f + '_mask'])
            v.append(row[f + '_delta'])
            v.append(row[f + '_outlier'])
        trajectory.append(v)
    return np.array(trajectory, dtype=np.float32)  # (T, 172)


def analyze_data_quality(df_prepared, feature_cols):
    """
    分析数据质量处理结果
    """
    print("\n=== 数据质量分析 ===")

    for f in feature_cols:
        mask_sum = df_prepared[f + '_mask'].sum()
        outlier_sum = df_prepared[f + '_outlier'].sum()
        inf_flag_sum = df_prepared[f + '_inf_flag'].sum() if f + '_inf_flag' in df_prepared.columns else 0

        total_points = len(df_prepared)
        missing_percentage = (total_points - mask_sum) / total_points * 100
        outlier_percentage = outlier_sum / total_points * 100

        print(f"特征: {f}")
        print(f"  - 缺失值: {total_points - mask_sum} ({missing_percentage:.2f}%)")
        print(f"  - 异常值: {outlier_sum} ({outlier_percentage:.2f}%)")
        print(f"  - 无穷大标记: {inf_flag_sum}")

if __name__ == "__main__":
    # 参数设置
    data_dir = "feature_all/unit15"
    feature_selected_dir = os.path.join(data_dir, "feature_selected")

    # 1. 读取原始数据
    original_data_path = os.path.join(data_dir, "15_feature_all_freq.csv")
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

    # 3. 应用完整预处理流水线
    print("\n" + "=" * 50)
    print("开始完整数据预处理流水线")
    print("=" * 50)

    df_enhanced, sequences = prepare_for_gru_complete(
        df_original,
        selected_features,
        time_col='cycle',
        seq_len=30,
        max_delta=30
    )

    # 4. 分析数据质量
    analyze_data_quality(df_enhanced, selected_features)

    # 5. 保存结果
    print("\n=== 保存结果 ===")

    # 保存增强后的DataFrame
    enhanced_data_path = os.path.join(feature_selected_dir, "enhanced_features_complete.csv")
    df_enhanced.to_csv(enhanced_data_path, index=False)
    print(f"✓ 完整增强数据已保存: {enhanced_data_path}")

    # 保存序列数据（滑动窗口）
    sequences_path = os.path.join(feature_selected_dir, "sequences_complete.npy")
    np.save(sequences_path, sequences)
    print(f"✓ 完整序列数据已保存: {sequences_path}")

    # 保存完整退化轨迹（用于维护决策，pointer 与真实退化阶段对应）
    trajectory = build_full_trajectory(df_enhanced, selected_features)
    trajectory_path = os.path.join(feature_selected_dir, "trajectory_complete.npy")
    np.save(trajectory_path, trajectory)
    print(f"✓ 完整退化轨迹已保存: {trajectory_path} (shape: {trajectory.shape})")

    # 保存质量分析报告
    quality_report = []
    for f in selected_features:
        mask_sum = df_enhanced[f + '_mask'].sum()
        outlier_sum = df_enhanced[f + '_outlier'].sum()
        inf_flag_sum = df_enhanced[f + '_inf_flag'].sum() if f + '_inf_flag' in df_enhanced.columns else 0

        total_points = len(df_enhanced)
        missing_percentage = (total_points - mask_sum) / total_points * 100
        outlier_percentage = outlier_sum / total_points * 100

        quality_report.append({
            'feature': f,
            'missing_count': total_points - mask_sum,
            'missing_percentage': missing_percentage,
            'outlier_count': outlier_sum,
            'outlier_percentage': outlier_percentage,
            'inf_flag_count': inf_flag_sum
        })

    quality_df = pd.DataFrame(quality_report)
    quality_path = os.path.join(feature_selected_dir, "quality_analysis.csv")
    quality_df.to_csv(quality_path, index=False)
    print(f"✓ 质量分析报告已保存: {quality_path}")

    print("\n" + "=" * 50)
    print("完整预处理流水线完成")
    print("=" * 50)
    print(f"最终序列形状: {sequences.shape}")
    print(f"样本数: {sequences.shape[0]}")
    print(f"序列长度: {sequences.shape[1]}")
    print(f"特征维度: {sequences.shape[2]}")
    print(f"每个原始特征扩展为4个分量:")
    print(f"  - 标准化数值")
    print(f"  - 缺失标志")
    print(f"  - 缺失时长")
    print(f"  - 异常标志")