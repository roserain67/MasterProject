"""
批量生成完整退化轨迹 trajectory_complete.npy
对 unit 14, 16, 18, 20 分别运行 step4.2 流水线并保存 trajectory_complete.npy
运行前需确保各 unit 已有 sequences_complete.npy（或本脚本会重新生成）
"""
import os
import sys
import numpy as np
import pandas as pd
import pickle

# 导入 step4.2 的函数
from importlib.util import spec_from_file_location, module_from_spec
_step42_path = os.path.join(os.path.dirname(__file__), "step4.2-质量增强与序列构建.py")
spec = spec_from_file_location("step42", _step42_path)
step42 = module_from_spec(spec)
spec.loader.exec_module(step42)

prepare_for_gru_complete = step42.prepare_for_gru_complete
build_full_trajectory = step42.build_full_trajectory

UNITS = [15]
FEATURE_ALL = "feature_all"


def process_unit(unit_id, base_dir="."):
    """对单个 unit 生成 trajectory_complete.npy"""
    data_dir = os.path.join(base_dir, FEATURE_ALL, f"unit{unit_id}")
    feature_selected_dir = os.path.join(data_dir, "feature_selected")
    csv_name = f"{unit_id}_feature_all_freq.csv"
    original_data_path = os.path.join(data_dir, csv_name)

    if not os.path.exists(original_data_path):
        print(f"  跳过 unit{unit_id}: 未找到 {original_data_path}")
        return False

    selected_features_path = os.path.join(feature_selected_dir, "selected_features.pkl")
    if not os.path.exists(selected_features_path):
        print(f"  跳过 unit{unit_id}: 未找到 selected_features.pkl")
        return False

    df_original = pd.read_csv(original_data_path)
    if df_original.index.name == 'cycle':
        df_original = df_original.reset_index()

    with open(selected_features_path, "rb") as f:
        selected_features = pickle.load(f)

    df_enhanced, sequences = prepare_for_gru_complete(
        df_original, selected_features,
        time_col='cycle', seq_len=30, max_delta=30
    )

    trajectory = build_full_trajectory(df_enhanced, selected_features)
    trajectory_path = os.path.join(feature_selected_dir, "trajectory_complete.npy")
    np.save(trajectory_path, trajectory)
    print(f"  ✓ unit{unit_id}: trajectory_complete.npy shape={trajectory.shape}")
    return True


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    print("批量生成完整退化轨迹...")
    for uid in UNITS:
        process_unit(uid, base_dir)
    print("完成。")
