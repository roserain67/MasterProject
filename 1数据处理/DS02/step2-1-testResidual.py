import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import torch

# ===================================
# 参数区
# ===================================
data_path = 'N-CMAPSS_DS02_selected.csv'
train_units = [16, 18, 20]
test_units = [14, 15]
cond_cols = ['alt', 'Mach', 'TRA', 'T2']
sensor_cols = ['Nf', 'Nc', 'T24', 'T48', 'P30', 'Wf']

# ===================================
# 1. 检查 GPU 环境
# ===================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 当前计算设备: {device}")

# ===================================
# 2. 加载数据
# ===================================
df = pd.read_csv(data_path)
print(f"数据形状: {df.shape}")
print(df.head())

# ===================================
# 3. 按 unit 分割训练 / 测试
# ===================================
train_df = df[df['unit'].isin(train_units)].reset_index(drop=True)
test_df = df[df['unit'].isin(test_units)].reset_index(drop=True)

X_train, y_train = train_df[cond_cols], train_df[sensor_cols]
X_test, y_test = test_df[cond_cols], test_df[sensor_cols]

# ===================================
# 4. 标准化工况特征
# ===================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===================================
# 5. GPU 模型训练（使用 XGBoost）
# ===================================
results = []
models = {}

for sensor in sensor_cols:
    print(f"\n🔹训练传感器: {sensor}")

    model = xgb.XGBRegressor(
        tree_method='gpu_hist' if device == "cuda" else 'hist',  # GPU模式
        predictor='gpu_predictor' if device == "cuda" else 'cpu_predictor',
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train_scaled, y_train[sensor])
    models[sensor] = model

    # 预测
    pred_train = model.predict(X_train_scaled)
    pred_test = model.predict(X_test_scaled)

    mae_train = mean_absolute_error(y_train[sensor], pred_train)
    mae_test = mean_absolute_error(y_test[sensor], pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train[sensor], pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test[sensor], pred_test))
    ratio = mae_test / mae_train

    results.append([sensor, mae_train, mae_test, rmse_train, rmse_test, ratio])

# ===================================
# 6. 结果汇总
# ===================================
res_df = pd.DataFrame(results, columns=['Sensor', 'MAE_train', 'MAE_test', 'RMSE_train', 'RMSE_test', 'MAE_ratio'])
print("\n=== 模型拟合结果 ===")
print(res_df)

# ===================================
# 7. 可视化 MAE 比例
# ===================================
plt.figure(figsize=(8,5))
sns.barplot(x='Sensor', y='MAE_ratio', data=res_df, color='steelblue')
plt.axhline(1.3, color='r', linestyle='--', label='阈值=1.3')
plt.title("MAE_test / MAE_train 比率 (越接近1越好)")
plt.legend()
plt.tight_layout()
plt.savefig('MAE_ratio_barplot_GPU.png', dpi=300)
plt.show()

# ===================================
# 8. 拟合效果散点（以 Nf 为例）
# ===================================
sensor = 'Nf'
model = models[sensor]
pred_test = model.predict(X_test_scaled)

plt.figure(figsize=(6,6))
plt.scatter(y_test[sensor], pred_test, s=10, alpha=0.6)
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title(f"传感器 {sensor} 拟合效果 (Test)")
plt.plot([y_test[sensor].min(), y_test[sensor].max()],
         [y_test[sensor].min(), y_test[sensor].max()],
         'r--')
plt.tight_layout()
plt.savefig(f'fit_scatter_{sensor}_GPU.png', dpi=300)
plt.show()

# ===================================
# 9. 建议判定
# ===================================
avg_ratio = res_df['MAE_ratio'].mean()
if avg_ratio < 1.3:
    print(f"\n✅ 平均 MAE_test/MAE_train = {avg_ratio:.2f} < 1.3，建议可以做残差提取（映射可迁移）")
elif avg_ratio < 1.6:
    print(f"\n⚠️ 平均 MAE_test/MAE_train = {avg_ratio:.2f}，映射一般，可考虑分组或正则化残差提取")
else:
    print(f"\n❌ 平均 MAE_test/MAE_train = {avg_ratio:.2f} > 1.6，不建议做残差提取（映射差异大）")
