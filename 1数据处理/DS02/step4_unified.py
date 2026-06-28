"""
统一数据生成脚本 — 替换 step4.1 + step4.2
确保所有 unit 使用相同的 43 个特征 + 全局统一标准化

运行方式（在 1数据处理/DS02/ 目录下）：
    python step4_unified.py

产出：
    feature_all/unified/
    ├── selected_features.pkl
    ├── global_scaler.pkl
    ├── feature_importance.csv
    └── unit{14,15,16,18,20}/
        └── feature_selected/
            ├── trajectory_complete.npy
            ├── enhanced_features_complete.csv
            └── quality_analysis.csv
"""
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

# ============================================================
# 参数
# ============================================================
UNITS = [14, 15, 16, 18, 20]

# 自动定位到 1数据处理/DS02/ 目录，无论从哪里运行
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

FEATURE_ALL = "feature_all"
OUTPUT_DIR = os.path.join(FEATURE_ALL, "unified")
K = 32  # 保留的传感器特征数量
CORR_THRESHOLD = 0.95
MAX_DELTA = 30

MANDATORY_SETTING = [
    "alt_mean", "alt_std",
    "Mach_mean", "Mach_std",
    "TRA_mean", "TRA_std",
    "T2_mean", "T2_std",
]
HEALTH_FEATURES = ["HPT_eff_mod_mean", "LPT_flow_mod_mean", "LPT_eff_mod_mean"]
MANDATORY_ALL = MANDATORY_SETTING + HEALTH_FEATURES


# ============================================================
# 阶段 1：读取所有 unit 数据 + 全局相关性去重
# ============================================================
def load_all_units():
    """读取所有 unit 的 feature_all_freq.csv，返回 {uid: DataFrame}"""
    unit_dfs = {}
    for uid in UNITS:
        csv_path = os.path.join(FEATURE_ALL, f"unit{uid}", f"{uid}_feature_all_freq.csv")
        df = pd.read_csv(csv_path, index_col="cycle")
        unit_dfs[uid] = df
        print(f"  unit{uid}: {df.shape[0]} rows, {df.shape[1]} cols")
    return unit_dfs


def global_correlation_filter(unit_dfs):
    """在合并数据上做相关性去重，返回统一的候选传感器特征列表。
    先排除在任何 unit 中 100% 为 NaN 的特征。"""
    # 排除在任一 unit 中全部缺失的特征
    all_cols = list(unit_dfs[UNITS[0]].columns)
    valid_cols = []
    for f in all_cols:
        if f in MANDATORY_ALL:
            continue
        all_present = True
        for uid, df in unit_dfs.items():
            if f not in df.columns or df[f].isna().all():
                all_present = False
                print(f"    排除 {f}: unit{uid} 中 100% NaN")
                break
        if all_present:
            valid_cols.append(f)

    print(f"  各 unit 均有效的传感器特征数: {len(valid_cols)}")

    merged = pd.concat(unit_dfs.values(), ignore_index=True)
    sensor_features = valid_cols
    print(f"  合并后传感器特征数: {len(sensor_features)}")

    if len(sensor_features) <= 1:
        return sensor_features

    corr_matrix = merged[sensor_features].corr().abs()
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > CORR_THRESHOLD:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                if merged[col_i].var() > merged[col_j].var():
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)

    filtered = [f for f in sensor_features if f not in to_drop]
    print(f"  相关性去重: 删除 {len(to_drop)} 个, 剩余 {len(filtered)} 个候选")
    return filtered


# ============================================================
# 阶段 2：统一 MI 特征选择
# ============================================================
def unified_mi_selection(unit_dfs, candidate_features):
    """对每个 unit 独立算 MI，取平均后选 top-K"""
    mi_per_unit = {}

    for uid, df in unit_dfs.items():
        available = [f for f in candidate_features if f in df.columns]
        health_available = [f for f in HEALTH_FEATURES if f in df.columns]

        if not available or not health_available:
            print(f"  unit{uid}: 跳过 MI（缺少特征）")
            continue

        X = df[available].fillna(df[available].median())
        if np.isinf(X.values).any():
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        mi_scores = np.zeros(len(available))
        for health_target in health_available:
            y = df[health_target].fillna(df[health_target].median())
            try:
                scores = mutual_info_regression(X_scaled, y, random_state=42)
                mi_scores += scores
            except Exception as e:
                print(f"  unit{uid}: MI 计算异常 ({health_target}): {e}")

        mi_scores /= len(health_available)
        mi_per_unit[uid] = pd.Series(mi_scores, index=available)
        print(f"  unit{uid}: MI 计算完成, top1={available[np.argmax(mi_scores)]} ({mi_scores.max():.4f})")

    all_features = sorted(set().union(*[set(s.index) for s in mi_per_unit.values()]))
    mi_avg = pd.Series(0.0, index=all_features)
    counts = pd.Series(0, index=all_features)
    for uid, scores in mi_per_unit.items():
        for f in scores.index:
            mi_avg[f] += scores[f]
            counts[f] += 1
    mi_avg = mi_avg / counts.clip(lower=1)
    mi_avg = mi_avg.sort_values(ascending=False)

    selected_sensor = mi_avg.head(K).index.tolist()
    selected_all = MANDATORY_ALL + selected_sensor

    importance_df = pd.DataFrame({"feature": mi_avg.index, "mi_avg": mi_avg.values})
    for uid, scores in mi_per_unit.items():
        importance_df[f"mi_unit{uid}"] = importance_df["feature"].map(scores).fillna(0)

    print(f"  统一选择: {len(selected_all)} 个特征 ({len(MANDATORY_ALL)} mandatory + {len(selected_sensor)} sensor)")
    return selected_all, importance_df


# ============================================================
# 阶段 3：全局标准化参数
# ============================================================
def compute_global_scaler(unit_dfs, selected_features):
    """合并所有 unit 数据，计算全局 mean/std"""
    all_data = []
    for uid, df in unit_dfs.items():
        cols = [f for f in selected_features if f in df.columns]
        sub = df[cols].copy()
        sub = sub.replace([np.inf, -np.inf], np.nan)
        all_data.append(sub)

    merged = pd.concat(all_data, ignore_index=True)

    global_mean = {}
    global_std = {}
    for f in selected_features:
        if f in merged.columns:
            vals = merged[f].dropna()
            global_mean[f] = float(vals.mean()) if len(vals) > 0 else 0.0
            std_val = float(vals.std()) if len(vals) > 1 else 1.0
            global_std[f] = std_val if std_val > 0 else 1.0
        else:
            global_mean[f] = 0.0
            global_std[f] = 1.0

    print(f"  全局 scaler: {len(global_mean)} 个特征")
    return global_mean, global_std


# ============================================================
# 阶段 4：逐 unit 生成 trajectory（使用全局 scaler）
# ============================================================
def prepare_for_gru_unified(features_df, feature_cols, global_mean, global_std,
                            time_col='cycle', max_delta=30):
    """
    与 step4.2 的 prepare_for_gru_complete 逻辑相同，
    但标准化使用全局 mean/std 而非 per-unit 统计量。
    """
    df = features_df.copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    # 1. 无穷大值处理
    for f in feature_cols:
        df[f + '_inf_flag'] = np.isinf(df[f]).astype(int)
        df[f] = df[f].replace([np.inf, -np.inf], np.nan)

    # 2. 计算 per-unit 中位数（用于缺失值填充）
    local_median = df[feature_cols].median()

    # 3. 逐时间点处理
    T = len(df)
    last_obs = {f: -9999 for f in feature_cols}
    processed_rows = []

    for t in range(T):
        row = df.iloc[t].copy()
        for f in feature_cols:
            val = df.at[t, f]
            if not pd.isna(val):
                row[f + '_mask'] = 1
                row[f + '_imputed'] = val
                last_obs[f] = t
            else:
                row[f + '_mask'] = 0
                if last_obs[f] == -9999:
                    row[f + '_imputed'] = local_median[f]
                    row[f + '_delta'] = max_delta
                else:
                    row[f + '_imputed'] = df.at[last_obs[f], f]
                    row[f + '_delta'] = min(t - last_obs[f], max_delta)

            if f + '_delta' not in row:
                row[f + '_delta'] = 0 if row[f + '_mask'] == 1 else max_delta

            z = (row[f + '_imputed'] - global_mean[f]) / (global_std[f] + 1e-12)
            row[f + '_outlier'] = 1 if abs(z) > 5 else 0

        processed_rows.append(row)

    df_prepared = pd.DataFrame(processed_rows)

    # 4. 标准化 — 使用全局 scaler
    for f in feature_cols:
        mu = global_mean[f]
        sigma = global_std[f]
        df_prepared[f + '_imputed'] = (df_prepared[f + '_imputed'] - mu) / sigma
        df_prepared[f + '_delta'] = df_prepared[f + '_delta'] / max_delta

    return df_prepared


def build_full_trajectory(df_prepared, feature_cols):
    """构建完整退化轨迹 (T, 43)，只保留标准化后的特征值"""
    T = len(df_prepared)
    trajectory = []
    for t in range(T):
        row = df_prepared.iloc[t]
        v = []
        for f in feature_cols:
            v.append(row[f + '_imputed'])
        trajectory.append(v)
    return np.array(trajectory, dtype=np.float32)


def analyze_data_quality(df_prepared, feature_cols):
    """生成质量分析报告"""
    report = []
    for f in feature_cols:
        mask_sum = df_prepared[f + '_mask'].sum()
        outlier_sum = df_prepared[f + '_outlier'].sum()
        inf_flag_sum = df_prepared[f + '_inf_flag'].sum() if f + '_inf_flag' in df_prepared.columns else 0
        total = len(df_prepared)
        report.append({
            'feature': f,
            'missing_count': total - mask_sum,
            'missing_pct': (total - mask_sum) / total * 100,
            'outlier_count': outlier_sum,
            'outlier_pct': outlier_sum / total * 100,
            'inf_count': inf_flag_sum,
        })
    return pd.DataFrame(report)


def generate_trajectories(unit_dfs, selected_features, global_mean, global_std):
    """对每个 unit 生成 trajectory_complete.npy"""
    results = {}
    for uid, df in unit_dfs.items():
        unit_dir = os.path.join(OUTPUT_DIR, f"unit{uid}", "feature_selected")
        os.makedirs(unit_dir, exist_ok=True)

        df_reset = df.reset_index() if df.index.name == 'cycle' else df.copy()

        df_prepared = prepare_for_gru_unified(
            df_reset, selected_features, global_mean, global_std,
            time_col='cycle', max_delta=MAX_DELTA
        )

        trajectory = build_full_trajectory(df_prepared, selected_features)
        traj_path = os.path.join(unit_dir, "trajectory_complete.npy")
        np.save(traj_path, trajectory)

        enhanced_path = os.path.join(unit_dir, "enhanced_features_complete.csv")
        df_prepared.to_csv(enhanced_path, index=False)

        quality = analyze_data_quality(df_prepared, selected_features)
        quality.to_csv(os.path.join(unit_dir, "quality_analysis.csv"), index=False)

        results[uid] = trajectory.shape
        print(f"  unit{uid}: trajectory shape={trajectory.shape}")

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("统一数据生成 (step4_unified)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 阶段 1
    print("\n[阶段1] 读取所有 unit 数据 + 全局相关性去重")
    unit_dfs = load_all_units()
    candidate_features = global_correlation_filter(unit_dfs)

    # 阶段 2
    print("\n[阶段2] 统一 MI 特征选择")
    selected_features, importance_df = unified_mi_selection(unit_dfs, candidate_features)

    with open(os.path.join(OUTPUT_DIR, "selected_features.pkl"), "wb") as f:
        pickle.dump(selected_features, f)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    print(f"  已保存: selected_features.pkl, feature_importance.csv")

    # 阶段 3
    print("\n[阶段3] 计算全局标准化参数")
    global_mean, global_std = compute_global_scaler(unit_dfs, selected_features)

    with open(os.path.join(OUTPUT_DIR, "global_scaler.pkl"), "wb") as f:
        pickle.dump({"mean": global_mean, "std": global_std}, f)
    print(f"  已保存: global_scaler.pkl")

    # 阶段 4
    print("\n[阶段4] 逐 unit 生成 trajectory")
    results = generate_trajectories(unit_dfs, selected_features, global_mean, global_std)

    # 汇总
    print("\n" + "=" * 60)
    print("完成！统一特征列表:")
    print(f"  mandatory ({len(MANDATORY_ALL)}): {MANDATORY_ALL}")
    print(f"  sensor ({len(selected_features) - len(MANDATORY_ALL)}): {selected_features[len(MANDATORY_ALL):]}")
    print(f"\n各 unit trajectory shape:")
    for uid, shape in results.items():
        print(f"  unit{uid}: {shape}")
    print(f"\n输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
