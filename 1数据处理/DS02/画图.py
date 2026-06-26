import os
import pandas as pd
import matplotlib.pyplot as plt
import math

# === 参数区 ===
file_path = 'D:\\yyo-Python\\0毕设\\数据处理\\DS02\\N-CMAPSS_DS02_selected.csv'  # 输入文件路径
save_dir = 'sensor_plots'  # 输出文件夹名称
os.makedirs(save_dir, exist_ok=True)

# === 1. 读取数据 ===
df = pd.read_csv(file_path)

# 检查列结构
print("列名:", df.columns.tolist())

# 假设格式为: unit, cycle, sensor1, sensor2, ...
unit_col = df.columns[0]
cycle_col = df.columns[1]
sensor_cols = df.columns[2:]

# 计算子图的行列数
n_sensors = len(sensor_cols)
n_cols = 4  # 每行显示4个子图
n_rows = math.ceil(n_sensors / n_cols)

# === 2. 按 unit 和 cycle 分组绘图 ===
for unit_id, unit_data in df.groupby(unit_col):
    for cycle_id, cycle_data in unit_data.groupby(cycle_col):
        # 创建大图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))

        # 如果只有一行，确保axes是二维数组
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # 在每个子图上绘制单个传感器参数
        for i, sensor in enumerate(sensor_cols):
            row = i // n_cols
            col = i % n_cols

            axes[row, col].plot(cycle_data.index, cycle_data[sensor], linewidth=1.5, color='blue')
            axes[row, col].set_title(f'{sensor}', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel('Sample Index', fontsize=10)
            axes[row, col].set_ylabel('Value', fontsize=10)
            axes[row, col].grid(True, alpha=0.3)

        # 隐藏多余的子图
        for i in range(n_sensors, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)

        # 设置总标题
        fig.suptitle(f'Unit {unit_id} - Cycle {cycle_id}', fontsize=16, fontweight='bold', y=0.98)

        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 为总标题留出空间

        # 保存图片
        save_path = os.path.join(save_dir, f'unit{unit_id}_cycle{cycle_id}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f" 已保存: {save_path}")

print("全部绘图完成！")