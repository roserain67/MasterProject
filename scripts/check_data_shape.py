"""检查各 unit 的 trajectory_complete.npy shape 和数值范围，结果写入 logs/gru_diagnostic/data_shape.txt"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.utils.paths import find_project_root
os.chdir(find_project_root())

BASE = "1数据处理/DS02/feature_all/unified"
UNITS = [14, 15, 16, 18, 20]
OUT_DIR = "logs/gru_diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

lines = []
for uid in UNITS:
    path = f"{BASE}/unit{uid}/trajectory_complete.npy"
    try:
        d = np.load(path)
        line = (f"unit{uid}: shape={d.shape}, dtype={d.dtype}, "
                f"min={d.min():.4f}, max={d.max():.4f}, "
                f"nan={np.isnan(d).sum()}, inf={np.isinf(d).sum()}")
    except Exception as e:
        line = f"unit{uid}: FAILED - {e}"
    print(line)
    lines.append(line)

out_path = os.path.join(OUT_DIR, "data_shape.txt")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved to {out_path}")
