"""参数扫描：对比不同 entropy_coef 的效果"""
import yaml
import sys
import os
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.paths import find_project_root
os.chdir(find_project_root())

from src.pearl import train

with open("configs/pearl_default.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

sweeps = [
    # 基线（之前已证明锁死太快）
    {"entropy_coef": 0.05, "entropy_coef_final": 0.02, "init_temp": 1.4, "temp_decay_ep": 300},
    # 当前默认
    {"entropy_coef": 0.08, "entropy_coef_final": 0.05, "init_temp": 2.0, "temp_decay_ep": 600},
    # 中等
    {"entropy_coef": 0.15, "entropy_coef_final": 0.08, "init_temp": 2.5, "temp_decay_ep": 500},
    # 激进
    {"entropy_coef": 0.25, "entropy_coef_final": 0.10, "init_temp": 3.0, "temp_decay_ep": 400},
    # 非常激进
    {"entropy_coef": 0.35, "entropy_coef_final": 0.15, "init_temp": 4.0, "temp_decay_ep": 500},
]

for i, params in enumerate(sweeps):
    cfg = copy.deepcopy(base_cfg)
    cfg.update(params)
    label = "_".join(f"{k}={v}" for k, v in params.items())
    cfg["log_dir"] = f"logs/sweep/{label}"
    cfg["num_episodes"] = 800

    print(f"\n{'='*60}")
    print(f"Sweep {i+1}/{len(sweeps)}: {params}")
    print(f"Log dir: {cfg['log_dir']}")
    print(f"{'='*60}\n")

    train(cfg)

print("\n全部完成！结果在 logs/sweep/ 下。")
