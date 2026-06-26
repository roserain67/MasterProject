"""
对比实验入口：同一批数据与 GRU，顺序训练 SAC / TD3 / TQC，各模型结果分别保存到 compare_dir/<model>/。
运行方式：在 3建模决策 目录下执行 python run_compare.py
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def load_sequences(data_base_path, unit_ids, use_trajectory=True):
    import numpy as np
    all_sequences = []
    for unit_id in unit_ids:
        traj_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "trajectory_complete.npy")
        seq_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "sequences_complete.npy")
        if use_trajectory and os.path.exists(traj_path):
            traj = np.load(traj_path)
            if traj.ndim == 3:
                traj = traj.squeeze(0)
            if traj.ndim == 2:
                all_sequences.append(traj)
        elif os.path.exists(seq_path):
            for s in np.load(seq_path):
                all_sequences.append(s)
    return all_sequences


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    data_base = os.path.join(base, "..", "1数据处理", "DS02", "feature_all")
    if not os.path.exists(data_base):
        data_base = os.path.join(base, "1数据处理", "DS02", "feature_all")

    train_sequences = load_sequences(data_base, [16, 18, 20])
    if not train_sequences:
        print("未找到训练数据，请检查路径:", data_base)
        sys.exit(1)

    test_14 = load_sequences(data_base, [14])
    test_15 = load_sequences(data_base, [15])
    test_by_unit = {
        14: test_14 if test_14 else train_sequences[: min(10, len(train_sequences))],
        15: test_15 if test_15 else test_14 if test_14 else train_sequences[: min(10, len(train_sequences))],
    }

    compare_dir = os.path.join(base, "..", "logs", "compare")
    os.makedirs(compare_dir, exist_ok=True)

    from SAC import load_gru_encoder, train as train_sac
    encoder = load_gru_encoder()

    num_episodes = 300

    print("===== 1/3 SAC =====")
    train_sac(
        encoder_model=encoder,
        train_sequences=train_sequences,
        test_by_unit=test_by_unit,
        log_dir=os.path.join(compare_dir, "SAC"),
        num_episodes=num_episodes,
    )

    print("===== 2/3 TD3 =====")
    from TD3 import train as train_td3
    train_td3(
        encoder_model=encoder,
        train_sequences=train_sequences,
        test_by_unit=test_by_unit,
        log_dir=os.path.join(compare_dir, "TD3"),
        num_episodes=num_episodes,
    )

    print("===== 3/3 TQC =====")
    from TQC import train as train_tqc
    train_tqc(
        encoder_model=encoder,
        train_sequences=train_sequences,
        test_by_unit=test_by_unit,
        log_dir=os.path.join(compare_dir, "TQC"),
        num_episodes=num_episodes,
    )

    print("全部模型训练完成。运行 plot_compare.py 生成对比图：")
    print("  python plot_compare.py")
    print("或指定目录： python plot_compare.py --dir logs/compare")


if __name__ == "__main__":
    main()
