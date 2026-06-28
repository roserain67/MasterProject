"""
GRU 编码器诊断：
1. 退化轨迹可视化（PCA + 逐步编码）
2. 冗余性检查（GRU输出 vs 归一化位置的线性回归 R²）
3. 区分能力检查（同一位置处不同 unit 的 GRU 输出是否可分）
4. 每一维的单调性检查

结果输出到 logs/gru_diagnostic/
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import os, sys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = "1数据处理/DS02/feature_all/unified"
PRETRAIN_PATH = "pretrain/gru_pretrained.pt"
UNITS = [14, 15, 16, 18, 20]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "logs/gru_diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []

def log(msg):
    print(msg)
    report_lines.append(msg)

# ---------- 加载 GRU ----------
sys.path.insert(0, ".")
from src.encoder import load_gru_encoder

encoder = load_gru_encoder(PRETRAIN_PATH).eval().to(DEVICE)
log(f"GRU loaded, device={DEVICE}")

# ---------- 逐步编码每条序列 ----------
unit_embeddings = {}

for uid in UNITS:
    path = f"{BASE}/unit{uid}/feature_selected/trajectory_complete.npy"
    data = np.load(path).astype(np.float32)
    if data.ndim == 2:
        data = data[np.newaxis, :]

    n_seqs = data.shape[0]
    records = []
    for si in range(n_seqs):
        seq = data[si]
        seq_len = seq.shape[0]
        for t in range(1, seq_len + 1):
            prefix = torch.tensor(seq[:t], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = encoder(prefix).squeeze(0).cpu().numpy()
            records.append((si, t, t / seq_len, emb))

    unit_embeddings[uid] = records
    log(f"unit{uid}: {n_seqs} seq(s), seq_len={data.shape[1]}, {len(records)} embeddings")

# ============================================================
# 诊断 1：PCA 轨迹可视化
# ============================================================
log("\n=== 诊断1: PCA 轨迹可视化 ===")

all_embs = []
all_labels = []
for uid, recs in unit_embeddings.items():
    for si, t, prog, emb in recs:
        all_embs.append(emb)
        all_labels.append((uid, si, prog))

all_embs = np.array(all_embs)
pca = PCA(n_components=3)
pca_result = pca.fit_transform(all_embs)
log(f"PCA explained variance: {pca.explained_variance_ratio_[:3].round(4)}")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(UNITS)))

ax = axes[0]
for i, uid in enumerate(UNITS):
    mask = [j for j, (u, s, p) in enumerate(all_labels) if u == uid]
    ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
              c=[colors[i]], s=5, alpha=0.3, label=f'unit{uid}')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('GRU embeddings: PC1 vs PC2')
ax.legend()

ax = axes[1]
for i, uid in enumerate(UNITS):
    indices = [j for j, (u, s, p) in enumerate(all_labels) if u == uid]
    progs = [all_labels[j][2] for j in indices]
    ax.scatter(progs, pca_result[indices, 0],
              c=[colors[i]], s=5, alpha=0.3, label=f'unit{uid}')
ax.set_xlabel('Progress (t / seq_len)')
ax.set_ylabel('PC1')
ax.set_title('PC1 vs normalized position')
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_trajectories.png"), dpi=200)
plt.close()
log("Saved: pca_trajectories.png")

# ============================================================
# 诊断 2：冗余性检查
# ============================================================
log("\n=== 诊断2: 冗余性检查 R² (GRU emb → progress) ===")

progs = np.array([p for u, s, p in all_labels]).reshape(-1, 1)

reg = LinearRegression().fit(all_embs, progs)
r2_global = r2_score(progs, reg.predict(all_embs))

r2_per_unit = {}
for uid in UNITS:
    mask = [j for j, (u, s, p) in enumerate(all_labels) if u == uid]
    X = all_embs[mask]
    y = progs[mask]
    reg_u = LinearRegression().fit(X, y)
    r2_per_unit[uid] = r2_score(y, reg_u.predict(X))

log(f"Global R²: {r2_global:.4f}")
for uid, r2 in r2_per_unit.items():
    log(f"  unit{uid}: R² = {r2:.4f}")

if r2_global > 0.95:
    log("结论: GRU 输出几乎等价于位置信息，64维没有额外价值")
elif r2_global > 0.7:
    log("结论: GRU 编码了位置信息 + 额外信号")
else:
    log("结论: GRU 输出与位置关系不大，可能是噪声或别的信号")

# ============================================================
# 诊断 3：区分能力检查
# ============================================================
log("\n=== 诊断3: 区分能力检查 (同一进度处不同 unit 是否可分) ===")

progress_bins = [0.25, 0.5, 0.75]

fig, axes = plt.subplots(1, len(progress_bins), figsize=(6 * len(progress_bins), 5))

for bi, target_prog in enumerate(progress_bins):
    unit_embs_at_prog = {}
    for uid in UNITS:
        embs = []
        for j, (u, s, p) in enumerate(all_labels):
            if u == uid and abs(p - target_prog) < 0.05:
                embs.append(all_embs[j])
        if len(embs) > 0:
            unit_embs_at_prog[uid] = np.array(embs)

    if len(unit_embs_at_prog) < 2:
        log(f"  progress={target_prog}: 数据不足，跳过")
        continue

    centroids = {u: e.mean(axis=0) for u, e in unit_embs_at_prog.items()}
    uids_present = list(centroids.keys())

    inter_dists = []
    for i in range(len(uids_present)):
        for j in range(i + 1, len(uids_present)):
            d = np.linalg.norm(centroids[uids_present[i]] - centroids[uids_present[j]])
            inter_dists.append(d)

    intra_dists = []
    for u, embs in unit_embs_at_prog.items():
        c = centroids[u]
        dists = np.linalg.norm(embs - c, axis=1)
        intra_dists.extend(dists.tolist())

    inter_mean = np.mean(inter_dists)
    intra_mean = np.mean(intra_dists) if intra_dists else 1e-8
    ratio = inter_mean / (intra_mean + 1e-8)

    verdict = '可区分' if ratio > 1.5 else '不可区分'
    log(f"  progress={target_prog}: inter={inter_mean:.4f}, intra={intra_mean:.4f}, "
        f"ratio={ratio:.2f} ({verdict})")

    ax = axes[bi]
    all_at_prog = np.vstack(list(unit_embs_at_prog.values()))
    pca_local = PCA(n_components=2).fit(all_at_prog)
    for i, uid in enumerate(uids_present):
        proj = pca_local.transform(unit_embs_at_prog[uid])
        ax.scatter(proj[:, 0], proj[:, 1], c=[colors[UNITS.index(uid)]],
                  s=20, alpha=0.6, label=f'unit{uid}')
    ax.set_title(f'progress={target_prog}, ratio={ratio:.2f}')
    ax.legend(fontsize=8)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "unit_discriminability.png"), dpi=200)
plt.close()
log("Saved: unit_discriminability.png")

# ============================================================
# 诊断 4：每一维的单调性检查
# ============================================================
log("\n=== 诊断4: 每维与时间的单调性 (Spearman |ρ|) ===")

dim = all_embs.shape[1]
mono_scores = np.zeros((len(UNITS), dim))

for ui, uid in enumerate(UNITS):
    recs = unit_embeddings[uid]
    seqs_dict = {}
    for si, t, prog, emb in recs:
        seqs_dict.setdefault(si, []).append((t, emb))

    dim_corrs = np.zeros(dim)
    n_seqs = 0
    for si, points in seqs_dict.items():
        points.sort(key=lambda x: x[0])
        ts = np.array([p[0] for p in points])
        embs = np.array([p[1] for p in points])
        if len(ts) < 5:
            continue
        n_seqs += 1
        for d in range(dim):
            corr, _ = spearmanr(ts, embs[:, d])
            dim_corrs[d] += abs(corr) if not np.isnan(corr) else 0
    if n_seqs > 0:
        mono_scores[ui] = dim_corrs / n_seqs

avg_mono = mono_scores.mean(axis=0)
high_mono_dims = int(np.sum(avg_mono > 0.7))
low_mono_dims = int(np.sum(avg_mono < 0.3))

log(f"64维中: 高单调性(|ρ|>0.7)={high_mono_dims}维, 低单调性(|ρ|<0.3)={low_mono_dims}维")
log(f"平均 |ρ|: {avg_mono.mean():.4f}")

plt.figure(figsize=(12, 4))
plt.bar(range(dim), avg_mono, color='steelblue', alpha=0.7)
plt.axhline(y=0.7, color='red', linestyle='--', label='高单调阈值 0.7')
plt.axhline(y=0.3, color='orange', linestyle='--', label='低单调阈值 0.3')
plt.xlabel('GRU embedding dimension')
plt.ylabel('Mean |Spearman ρ| with time')
plt.title('每维与时间的单调性')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "monotonicity.png"), dpi=200)
plt.close()
log("Saved: monotonicity.png")

# ============================================================
# 汇总
# ============================================================
log("\n" + "=" * 60)
log("诊断汇总")
log("=" * 60)
log(f"PCA explained variance (前3): {pca.explained_variance_ratio_[:3].round(4)}")
log(f"R² (emb→progress): {r2_global:.4f}")
log(f"高单调维度数: {high_mono_dims}/64")

if r2_global > 0.95 and high_mono_dims > 50:
    log("\n判定: GRU 本质上只编码了位置信息，64维是冗余的")
    log("建议: 去掉 GRU 或降维到 2-4 维，先验证 RL 训练")
elif r2_global > 0.7:
    log("\n判定: GRU 编码了位置 + 部分额外信号")
    log("建议: 保留 GRU 但需要统一特征后重新 pretrain 再判断")
else:
    log("\n判定: GRU 输出与位置关系弱")
    log("建议: 可能编码了有价值的非线性特征，也可能是噪声")
    log("       需要统一特征后重新 pretrain 才能确定")

# ---------- 保存报告 ----------
report_path = os.path.join(OUT_DIR, "gru_diagnostic_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")
log(f"\n报告已保存到 {report_path}")
