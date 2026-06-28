import os
import numpy as np


def load_sequences(data_base_path, unit_ids, use_trajectory=True):
    """
    加载序列数据，返回 (sequences, unit_ids)。
    优先使用 trajectory_complete.npy，否则回退到 sequences_complete.npy。
    """
    all_sequences = []
    all_unit_ids = []
    for unit_id in unit_ids:
        traj_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "trajectory_complete.npy")
        traj_path_unified = os.path.join(data_base_path, f"unit{unit_id}", "trajectory_complete.npy")
        seq_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "sequences_complete.npy")
        if use_trajectory and os.path.exists(traj_path):
            traj = np.load(traj_path)
        elif use_trajectory and os.path.exists(traj_path_unified):
            traj = np.load(traj_path_unified)
        else:
            traj = None

        if traj is not None:
            if traj.ndim == 3:
                traj = traj.squeeze(0)
            if traj.ndim == 2:
                all_sequences.append(traj)
                all_unit_ids.append(unit_id)
        elif os.path.exists(seq_path):
            seqs = np.load(seq_path)
            for i in range(len(seqs)):
                all_sequences.append(seqs[i])
                all_unit_ids.append(unit_id)
    return all_sequences, all_unit_ids
