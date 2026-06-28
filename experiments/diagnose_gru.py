"""
GRU 编码器诊断脚本
回答核心问题：64维 GRU 表征是否比 pos_A/pos_B 多提供了有用信息？

检查项:
  1. 退化轨迹可视化 (PCA 2D) — 不同 unit 轨迹是否分离
  2. 位置预测 R² — GRU 输出与归一化位置的冗余度
  3. unit 区分能力 — 相同退化进度下，不同 unit 的嵌入是否可分
  4. 各维度信噪比 — 64 维中有多少维携带有用信号

输出: logs/diagnose_gru/ 下的图表和数值报告
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoder import load_gru_encoder
from src.utils.data_loader import load_sequences

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "logs/diagnose_gru"


def extract_embeddings(encoder, sequences, unit_ids):
    """对每条序列逐步截取前缀，得到 GRU 嵌入轨迹."""
    records = []
    encoder.eval()
    with torch.no_grad():
        for seq, uid in zip(sequences, unit_ids):
            seq_len = len(seq)
            step = max(1, seq_len // 40)
            for t in range(0, seq_len, step):
                prefix = seq[:t + 1].astype(np.float32)
                x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                emb = encoder(x).squeeze(0).cpu().numpy()
                records.append({
                    'unit_id': uid,
                    'timestep': t,
                    'progress': t / max(1, seq_len - 1),
                    'emb': emb,
                })
    return records


def check1_trajectory_pca(records):
    """PCA 降维看不同 unit 的退化轨迹是否分离."""
    embs = np.array([r['emb'] for r in records])
    unit_ids = np.array([r['unit_id'] for r in records])
    progress = np.array([r['progress'] for r in records])

    pca = PCA(n_components=2)
    embs_2d = pca.fit_transform(embs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    unique_units = sorted(set(unit_ids))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_units)))

    ax = axes[0]
    for i, uid in enumerate(unique_units):
        mask = unit_ids == uid
        ax.plot(embs_2d[mask, 0], embs_2d[mask, 1], 'o-', color=colors[i],
                label=f'unit{uid}', markersize=3, alpha=0.7)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('GRU嵌入轨迹 (PCA 2D)')
    ax.legend()

    ax = axes[1]
    sc = ax.scatter(embs_2d[:, 0], embs_2d[:, 1], c=progress, cmap='RdYlGn_r',
                    s=15, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='退化进度 (0=健康, 1=故障)')
    ax.set_xlabel(f'PC1')
    ax.set_ylabel(f'PC2')
    ax.set_title('GRU嵌入 按退化进度着色')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'check1_trajectory_pca.png'), dpi=150)
    plt.close()

    var_explained = pca.explained_variance_ratio_[:2].sum()
    print(f"[Check1] PCA 前2维解释方差: {var_explained:.1%}")
    return pca


def check2_position_r2(records):
    """线性回归：GRU 64维 → 归一化位置. 检查冗余度."""
    embs = np.array([r['emb'] for r in records])
    progress = np.array([r['progress'] for r in records])

    reg = LinearRegression().fit(embs, progress)
    pred = reg.predict(embs)
    r2 = r2_score(progress, pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(progress, pred, s=5, alpha=0.3)
    ax.plot([0, 1], [0, 1], 'r--', label='完美预测')
    ax.set_xlabel('真实进度')
    ax.set_ylabel('从GRU嵌入预测的进度')
    ax.set_title(f'位置预测 R² = {r2:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'check2_position_r2.png'), dpi=150)
    plt.close()

    print(f"[Check2] 线性回归 R² = {r2:.4f}")
    if r2 > 0.95:
        print("  → GRU 输出与位置高度冗余，需要 Check3 判断是否有额外信息")
    elif r2 > 0.7:
        print("  → GRU 编码了位置信息但不完全，可能还有其他信号")
    else:
        print("  → GRU 输出与位置关系弱，可能编码了别的特征或是噪声")
    return r2


def check3_unit_discriminability(records):
    """相同退化进度下，不同 unit 的 GRU 嵌入是否可区分."""
    progress_bins = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
    results = {}

    fig, axes = plt.subplots(1, len(progress_bins), figsize=(6 * len(progress_bins), 5))

    for idx, (lo, hi) in enumerate(progress_bins):
        subset = [r for r in records if lo <= r['progress'] < hi]
        if len(subset) < 10:
            print(f"  进度区间 [{lo:.1f}, {hi:.1f}): 样本不足 ({len(subset)})")
            continue

        embs = np.array([r['emb'] for r in subset])
        labels = np.array([r['unit_id'] for r in subset])
        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            print(f"  进度区间 [{lo:.1f}, {hi:.1f}): 只有1个unit，跳过")
            continue

        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(embs)

        ax = axes[idx]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for i, uid in enumerate(unique_labels):
            mask = labels == uid
            ax.scatter(embs_2d[mask, 0], embs_2d[mask, 1], color=colors[i],
                       label=f'unit{uid}', s=30, alpha=0.7)
        ax.set_title(f'进度 [{lo:.1f}, {hi:.1f})')
        ax.legend(fontsize=8)

        if len(set(labels)) >= 2 and len(labels) >= len(set(labels)) + 1:
            sil = silhouette_score(embs, labels)
            results[f'{lo:.1f}-{hi:.1f}'] = sil
            ax.set_xlabel(f'silhouette = {sil:.3f}')

    plt.suptitle('相同进度区间下的 unit 区分能力')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'check3_unit_discriminability.png'), dpi=150)
    plt.close()

    for k, v in results.items():
        quality = "好" if v > 0.3 else ("一般" if v > 0.1 else "差")
        print(f"[Check3] 进度 {k}: silhouette = {v:.3f} ({quality})")
    return results


def check4_dimension_snr(records):
    """各维度信噪比：哪些维随退化单调变化，哪些是噪声."""
    embs = np.array([r['emb'] for r in records])
    progress = np.array([r['progress'] for r in records])

    correlations = []
    for d in range(embs.shape[1]):
        corr = np.corrcoef(embs[:, d], progress)[0, 1]
        correlations.append(corr)
    correlations = np.array(correlations)

    sorted_idx = np.argsort(np.abs(correlations))[::-1]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.bar(range(64), np.abs(correlations[sorted_idx]), color='steelblue', alpha=0.7)
    ax.axhline(y=0.3, color='r', linestyle='--', label='|corr| = 0.3 阈值')
    ax.set_xlabel('维度 (按|相关性|排序)')
    ax.set_ylabel('|与退化进度的相关系数|')
    ax.set_title('各维度与退化进度的相关性')
    ax.legend()

    n_useful = np.sum(np.abs(correlations) > 0.3)
    n_strong = np.sum(np.abs(correlations) > 0.6)

    ax = axes[1]
    top_dims = sorted_idx[:4]
    bottom_dims = sorted_idx[-2:]
    plot_dims = list(top_dims) + list(bottom_dims)

    for d in plot_dims:
        label = f'dim{d} (r={correlations[d]:.2f})'
        ax.scatter(progress, embs[:, d], s=3, alpha=0.3, label=label)
    ax.set_xlabel('退化进度')
    ax.set_ylabel('GRU输出值')
    ax.set_title('代表性维度随退化进度的变化')
    ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'check4_dimension_snr.png'), dpi=150)
    plt.close()

    print(f"[Check4] |corr| > 0.6 的维度: {n_strong}/64")
    print(f"[Check4] |corr| > 0.3 的维度: {n_useful}/64")
    print(f"[Check4] 前5强维度: {sorted_idx[:5]} (corr = {correlations[sorted_idx[:5]]})")
    return correlations


def print_summary(r2, sil_results, correlations):
    """综合判断."""
    print("\n" + "=" * 60)
    print("GRU 诊断总结")
    print("=" * 60)

    n_useful = np.sum(np.abs(correlations) > 0.3)
    avg_sil = np.mean(list(sil_results.values())) if sil_results else 0

    print(f"  位置预测 R²:     {r2:.4f}")
    print(f"  有效维度数:       {n_useful}/64 (|corr|>0.3)")
    print(f"  平均 silhouette:  {avg_sil:.3f}")

    print()
    if r2 > 0.95 and avg_sil > 0.3:
        print("  判断: GRU 编码了位置 + unit 差异信息，对 PEARL 有价值")
        print("  建议: 保留 GRU，问题在 RL 训练侧")
    elif r2 > 0.95 and avg_sil <= 0.3:
        print("  判断: GRU 主要编码了位置，对 PEARL 元学习帮助有限")
        print("  建议: 可以考虑降维或简化 state，优先调 RL")
    elif r2 <= 0.7 and avg_sil <= 0.1:
        print("  判断: GRU 输出质量差，64维可能是噪声")
        print("  建议: 冻结/去掉 GRU，用 pos 训练；或重新 pretrain")
    else:
        print("  判断: GRU 编码了位置之外的信息，需人工进一步判断是否有用")
        print("  建议: 保留 GRU，同时调 RL 训练侧")
    print("=" * 60)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    encoder = load_gru_encoder("pretrain/gru_pretrained.pt")

    all_units = [14, 15, 16, 18, 20]
    sequences, unit_ids = load_sequences("1数据处理/DS02/feature_all", all_units)
    print(f"加载了 {len(sequences)} 条序列, units: {sorted(set(unit_ids))}")
    for uid in sorted(set(unit_ids)):
        idx = [i for i, u in enumerate(unit_ids) if u == uid]
        lens = [len(sequences[i]) for i in idx]
        print(f"  unit{uid}: {len(idx)} 条, 长度 {lens}")

    print("\n提取 GRU 嵌入...")
    records = extract_embeddings(encoder, sequences, unit_ids)
    print(f"共 {len(records)} 个嵌入样本")

    print("\n--- Check 1: 退化轨迹 PCA 可视化 ---")
    check1_trajectory_pca(records)

    print("\n--- Check 2: 位置预测 R² ---")
    r2 = check2_position_r2(records)

    print("\n--- Check 3: Unit 区分能力 ---")
    sil_results = check3_unit_discriminability(records)

    print("\n--- Check 4: 各维度信噪比 ---")
    correlations = check4_dimension_snr(records)

    print_summary(r2, sil_results, correlations)


if __name__ == "__main__":
    main()
