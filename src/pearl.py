import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.env import MaintenanceEnv
from src.encoder import GRUEncoder, load_gru_encoder
from src.utils.replay_buffer import EpisodeReplayBuffer, ContextBuffer
from src.utils.data_loader import load_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """统一设置 random / numpy / torch 随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================
# PEARL 网络组件
# ======================================================
class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden=128, z_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 5)
        return mu, logvar


def sample_z(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


class Actor(nn.Module):
    def __init__(self, state_dim, z_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        return F.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, z_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, s, z):
        x = torch.cat([s, z], dim=-1)
        return self.net(x)


# ======================================================
# 训练
# ======================================================
def train(cfg):
    """PEARL 训练主函数，所有超参从 cfg dict 传入"""
    # ---------- 随机种子（必须在建网络/采样之前）----------
    seed = cfg.get("seed")
    if seed is not None:
        set_seed(seed)
        print(f">> 随机种子已设置: {seed}")

    # ---------- 数据加载 ----------
    data_base = cfg["data_base"]
    train_sequences, train_unit_ids = load_sequences(data_base, cfg["train_units"])
    if len(train_sequences) == 0:
        raise ValueError(f"未找到训练数据，请检查路径: {data_base}")

    test_by_unit = {}
    for uid in cfg["test_units"]:
        seqs, _ = load_sequences(data_base, [uid])
        if len(seqs) > 0:
            test_by_unit[uid] = seqs
    if not test_by_unit:
        test_by_unit[cfg["test_units"][0]] = train_sequences[:min(10, len(train_sequences))]

    # ---------- 模型 ----------
    no_gru = cfg.get("no_gru", False)
    if no_gru:
        encoder_model = None
    else:
        encoder_model = load_gru_encoder(cfg.get("pretrain_path"))
    state_dim = cfg["state_dim"]
    n_actions = cfg["n_actions"]
    z_dim = cfg["z_dim"]

    context_input_dim = state_dim + 2 + state_dim
    context_encoder = ContextEncoder(context_input_dim, z_dim=z_dim).to(DEVICE)
    actor = Actor(state_dim, z_dim, n_actions).to(DEVICE)
    critic = Critic(state_dim, z_dim, n_actions).to(DEVICE)
    critic_target = Critic(state_dim, z_dim, n_actions).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    opt_enc = optim.Adam(context_encoder.parameters(), lr=cfg["lr_encoder"])
    opt_actor = optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])

    target_entropy = 0.1 * np.log(n_actions)
    log_alpha = torch.tensor(np.log(cfg["entropy_coef"]), dtype=torch.float32, device=DEVICE, requires_grad=True)
    opt_alpha = optim.Adam([log_alpha], lr=cfg["lr_actor"])

    replay = EpisodeReplayBuffer(
        train_units=cfg["train_units"],
        good_capacity=cfg["good_capacity"],
        recent_capacity=cfg["recent_capacity"],
        context_k=cfg["context_k"],
        n_step=cfg.get("n_step", 1),
        gamma=cfg["gamma"]
    )
    ctxbuf = ContextBuffer(context_k=cfg["context_k"])

    # ---------- 超参 ----------
    num_episodes = cfg["num_episodes"]
    max_len = cfg["max_len"]
    gamma = cfg["gamma"]
    batch_size = cfg["batch_size"]
    init_temp = cfg["init_temp"]
    temp_decay_ep = cfg["temp_decay_ep"]
    tau = cfg["tau"]
    reward_clip = cfg["reward_clip"]
    td_clip = cfg["td_clip"]
    grad_clip = cfg["grad_clip"]
    warmup_ep = cfg["warmup_ep"]
    replay_good_thresh = cfg["replay_good_thresh"]
    replay_min_thresh = cfg["replay_min_thresh"]
    curriculum_ep = cfg["curriculum_ep"]

    n_step = cfg.get("n_step", 1)
    gamma_n = gamma ** n_step

    # ---------- 日志 ----------
    log_dir = cfg["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    records = []
    losses_actor = []
    losses_critic = []
    total_steps = 0
    best_reward = -np.inf

    # ---------- 诊断探针：固定参考态，跨 episode 可比 ----------
    # 用第一条训练序列的初始 state 作为固定锚点，z 取 zero-context 的 mu（确定性，稳定可比）
    ref_env = MaintenanceEnv(train_sequences[0], encoder_model, state_dim=state_dim, no_gru=no_gru)
    s_ref = torch.tensor(ref_env.reset(), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    zero_ctx = torch.zeros((1, context_input_dim), device=DEVICE)
    diag_records = []

    # ---------- 训练循环 ----------
    for ep in range(1, num_episodes + 1):
        ctxbuf.clear()
        temp = max(1.0, init_temp - (init_temp - 1.0) * max(0, ep - temp_decay_ep) / max(1, num_episodes - temp_decay_ep))

        idx = random.randint(0, len(train_sequences) - 1)
        seq = train_sequences[idx]
        unit_id = train_unit_ids[idx]
        env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim, no_gru=no_gru)
        s = env.reset()
        episode_reward = 0
        episode_actions = []
        ep_transitions = []
        ep_entropies = []
        ep_z_norms = []
        la_start = len(losses_actor)   # 本回合 loss 切片起点
        lc_start = len(losses_critic)

        for t in range(max_len):
            total_steps += 1
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            ctx = ctxbuf.sample_context()
            if ctx is None:
                ctx_t = torch.zeros((1, context_input_dim), device=DEVICE)
            else:
                ctx_t = torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            mu, logvar = context_encoder(ctx_t)
            z = sample_z(mu, logvar)

            with torch.no_grad():
                logits = actor.net(torch.cat([s_t, z], dim=-1))
                probs = F.softmax(logits / temp, dim=-1).cpu().numpy().flatten()
            probs = probs / probs.sum()

            eps_greedy = max(0.0, 0.3 * (1 - ep / 200))
            if np.random.rand() < eps_greedy:
                action = np.random.randint(n_actions)
            else:
                action = np.random.choice(n_actions, p=probs)
            ep_entropies.append(-np.sum(probs * np.log(probs + 1e-8)))
            ep_z_norms.append(float(torch.norm(z).item()))

            s2, r, done, _ = env.step(action)
            r_clipped = np.clip(r, -reward_clip, reward_clip)
            ep_transitions.append((s.copy(), action, r_clipped, s2.copy(), done))
            ctxbuf.push(s, action, r_clipped, s2, done)

            episode_reward += r
            episode_actions.append(action)
            s = s2

            # ---------- 网络更新 ----------
            sample_result = replay.sample(batch_size) if len(replay) > 0 else None
            if sample_result is not None:
                bs, ba, br, bs2, bd, ctx_list = sample_result
                n_batch = len(bs)
                if n_batch >= 32:
                    bs = torch.tensor(bs, dtype=torch.float32, device=DEVICE)
                    ba = torch.tensor(ba, dtype=torch.long, device=DEVICE)
                    br = torch.tensor(br, dtype=torch.float32, device=DEVICE)
                    bs2 = torch.tensor(bs2, dtype=torch.float32, device=DEVICE)
                    bd = torch.tensor(bd, dtype=torch.float32, device=DEVICE)

                    z_parts, mu_list, logvar_list = [], [], []
                    for ctx_vec, n in ctx_list:
                        ctx_b = torch.tensor(ctx_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        mu_b, logvar_b = context_encoder(ctx_b)
                        z_b = sample_z(mu_b, logvar_b).repeat(n, 1)
                        z_parts.append(z_b)
                        mu_list.append(mu_b)
                        logvar_list.append(logvar_b)
                    z_b = torch.cat(z_parts, dim=0)
                    mu_b = torch.cat(mu_list, dim=0)
                    logvar_b = torch.cat(logvar_list, dim=0)

                    z_b_for_critic = z_b.detach()

                    q = critic(bs, z_b_for_critic)
                    q_a = q.gather(1, ba.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        q_next = critic_target(bs2, z_b_for_critic).max(1)[0]
                        br_clipped = torch.clamp(br, -reward_clip, reward_clip)
                        td_target = br_clipped + gamma_n * (1 - bd) * q_next
                        td_target = torch.clamp(td_target, -td_clip, td_clip)

                    loss_c = F.smooth_l1_loss(q_a, td_target, beta=50.0)

                    alpha = log_alpha.exp().detach()
                    q_all = critic(bs, z_b_for_critic).detach()
                    probs_b = actor(bs, z_b)
                    log_probs = torch.log(probs_b + 1e-8)
                    loss_a = (probs_b * (alpha * log_probs - q_all)).sum(dim=1).mean()

                    kl = -0.5 * (1 + logvar_b - mu_b.pow(2) - logvar_b.exp()).mean()
                    reg = 0.1 * (mu_b.pow(2).mean() + logvar_b.exp().mean()) + 0.2 * kl

                    opt_critic.zero_grad()
                    loss_c.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                    opt_critic.step()
                    for p, pt in zip(critic.parameters(), critic_target.parameters()):
                        pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

                    opt_actor.zero_grad()
                    opt_enc.zero_grad()
                    (loss_a + reg).backward()
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(context_encoder.parameters()), grad_clip)
                    opt_actor.step()
                    opt_enc.step()

                    entropy = -(probs_b.detach() * log_probs.detach()).sum(dim=1).mean()
                    loss_alpha = (log_alpha * (entropy - target_entropy).detach()).mean()
                    opt_alpha.zero_grad()
                    loss_alpha.backward()
                    opt_alpha.step()

                    losses_actor.append(loss_a.item())
                    losses_critic.append(loss_c.item())

            if done:
                break

        min_th = -9999 if ep <= warmup_ep else replay_min_thresh
        replay.push_episode(ep_transitions, episode_reward, unit_id, replay_good_thresh, min_th)

        mean_entropy = float(np.mean(ep_entropies)) if ep_entropies else 0.0
        mean_z_norm = float(np.mean(ep_z_norms)) if ep_z_norms else 0.0

        # ---------- 诊断探针：固定参考态下测量策略/critic 状态 ----------
        with torch.no_grad():
            mu_ref, _ = context_encoder(zero_ctx)
            z_ref = mu_ref                       # 确定性 z，去掉采样噪声
            probs_ref = actor(s_ref, z_ref).cpu().numpy().flatten()
            q_ref = critic(s_ref, z_ref).cpu().numpy().flatten()
        ent_ref = float(-np.sum(probs_ref * np.log(probs_ref + 1e-8)))
        cur_alpha = float(log_alpha.exp().item())
        ep_la = float(np.mean(losses_actor[la_start:])) if len(losses_actor) > la_start else float("nan")
        ep_lc = float(np.mean(losses_critic[lc_start:])) if len(losses_critic) > lc_start else float("nan")
        diag_records.append({
            "episode": ep,
            "reward": episode_reward,
            "alpha": cur_alpha,            # auto-entropy 温度（看是否暴涨/卡死）
            "ent_ref": ent_ref,            # 固定态策略熵：→0 即真坍缩，max=ln5≈1.609
            "ent_ep": mean_entropy,        # 本回合实际动作熵
            "argmax_ref": int(np.argmax(probs_ref)),  # 固定态贪婪动作（对应 deterministic eval）
            **{f"p{i}": float(probs_ref[i]) for i in range(n_actions)},  # 各动作概率
            **{f"q{i}": float(q_ref[i]) for i in range(n_actions)},      # 各动作 Q（看高估/发散）
            "z_norm": mean_z_norm,
            "loss_actor": ep_la,
            "loss_critic": ep_lc,
        })

        records.append({
            "episode": ep, "reward": episode_reward,
            "action_entropy": mean_entropy, "z_norm": mean_z_norm,
            "actions": Counter(episode_actions)
        })

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_path = os.path.join(log_dir, "best_model.pt")
            torch.save({
                "actor": actor.state_dict(),
                "encoder": context_encoder.state_dict(),
                "critic": critic.state_dict(),
            }, best_path)

        if ep % 10 == 0:
            ps = " ".join(f"{probs_ref[i]:.2f}" for i in range(n_actions))
            qs = " ".join(f"{q_ref[i]:.0f}" for i in range(n_actions))
            print(f"Ep {ep}: reward={episode_reward:7.1f} | alpha={cur_alpha:.3f} ent_ref={ent_ref:.3f} "
                  f"argmax={int(np.argmax(probs_ref))} | p=[{ps}] Q=[{qs}] | z={mean_z_norm:.2f}")

    # ---------- 保存 CSV ----------
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "loss_reward_action.csv"), index=False)
    pd.DataFrame({"critic": losses_critic, "actor": losses_actor}).to_csv(os.path.join(log_dir, "loss_curves.csv"), index=False)
    pd.DataFrame(diag_records).to_csv(os.path.join(log_dir, "diagnostics.csv"), index=False)

    # ---------- 训练曲线图 ----------
    _plot_training(df, losses_critic, losses_actor, log_dir)

    print(f"训练完成！日志已保存到 {log_dir}")

    # ---------- 加载 best model 用于测试 ----------
    best_path = os.path.join(log_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        actor.load_state_dict(ckpt["actor"])
        context_encoder.load_state_dict(ckpt["encoder"])
        print(f"已加载 best model (best_reward={best_reward:.2f})")

    # ---------- 测试评估 ----------
    test_episodes = cfg.get("test_episodes", 300)
    deterministic = cfg.get("deterministic_eval", True)
    for unit_id, test_seqs in test_by_unit.items():
        test_results = evaluate(encoder_model, actor, context_encoder, test_seqs,
                                state_dim, n_actions, context_input_dim,
                                num_episodes=test_episodes, deterministic=deterministic,
                                no_gru=no_gru)
        _save_test_results(test_results, unit_id, log_dir)
    print(f"全部测试结果已保存到 {log_dir}")


# ======================================================
# 评估
# ======================================================
def evaluate(encoder_model, actor, context_encoder, test_sequences,
             state_dim, n_actions, context_input_dim,
             num_episodes=300, deterministic=True, no_gru=False):
    actor.eval()
    context_encoder.eval()

    test_results = []
    ctxbuf = ContextBuffer()

    for _ in range(min(3, len(test_sequences))):
        seq = random.choice(test_sequences)
        env_pre = MaintenanceEnv(seq, encoder_model, state_dim=state_dim, no_gru=no_gru)
        s = env_pre.reset()
        for _ in range(min(20, len(seq))):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            ctx = ctxbuf.sample_context()
            ctx_t = torch.zeros((1, context_input_dim), device=DEVICE) if ctx is None else torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mu, logvar = context_encoder(ctx_t)
                z = sample_z(mu, logvar)
                probs = actor(s_t, z).cpu().numpy().flatten()
            probs = probs / (probs.sum() + 1e-8)
            a = np.argmax(probs) if deterministic else np.random.choice(n_actions, p=probs)
            s2, r, done, _ = env_pre.step(a)
            ctxbuf.push(s, a, r, s2, done)
            s = s2
            if done:
                break

    for ep_idx in range(num_episodes):
        seq = random.choice(test_sequences)
        env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim, no_gru=no_gru)
        s = env.reset()
        episode_reward = 0
        episode_actions = []

        for t in range(200):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            ctx = ctxbuf.sample_context()
            ctx_t = torch.zeros((1, context_input_dim), device=DEVICE) if ctx is None else torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mu, logvar = context_encoder(ctx_t)
                z = sample_z(mu, logvar)
                probs = actor(s_t, z).cpu().numpy().flatten()
            probs = probs / (probs.sum() + 1e-8)
            action = np.argmax(probs) if deterministic else np.random.choice(n_actions, p=probs)
            s2, r, done, _ = env.step(action)
            ctxbuf.push(s, action, r, s2, done)
            episode_reward += r
            episode_actions.append(action)
            s = s2
            if done:
                break

        test_results.append({
            'episode': ep_idx + 1,
            'reward': episode_reward,
            'actions': dict(Counter(episode_actions))
        })

    actor.train()
    context_encoder.train()
    return test_results


# ======================================================
# 绘图辅助
# ======================================================
def _plot_training(df, losses_critic, losses_actor, log_dir, smooth_win=50):
    rewards = df["reward"].values
    r_smooth = pd.Series(rewards).rolling(smooth_win, min_periods=1).mean()
    r_std = pd.Series(rewards).rolling(smooth_win, min_periods=1).std().fillna(0)
    x = np.arange(len(rewards))

    plt.figure()
    plt.plot(x, rewards, color='steelblue', alpha=0.35, linewidth=0.8, label='原始值')
    plt.fill_between(x, r_smooth - r_std, r_smooth + r_std, color='steelblue', alpha=0.2)
    plt.plot(x, r_smooth, color='darkblue', linewidth=2, label='平滑曲线')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('PEARL Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "episode_reward.png"))
    plt.close()

    if losses_critic:
        lc = pd.Series(losses_critic)
        la = pd.Series(losses_actor)
        x_loss = np.arange(len(losses_critic))
        plt.figure()
        plt.plot(x_loss, lc.rolling(smooth_win, min_periods=1).mean(), color='darkred', linewidth=1.5, label='critic')
        plt.plot(x_loss, la.rolling(smooth_win, min_periods=1).mean(), color='darkgreen', linewidth=1.5, label='actor')
        plt.legend()
        plt.xlabel('更新步数')
        plt.ylabel('Loss')
        plt.title('PEARL Loss Curves')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(log_dir, "loss_curves.png"))
        plt.close()


def _save_test_results(test_results, unit_id, log_dir, smooth_win=10):
    suffix = f"unit{unit_id}"
    test_df = pd.DataFrame([{
        'episode': r['episode'], 'reward': r['reward'],
        **{f'action_{k}': v for k, v in r['actions'].items()}
    } for r in test_results])
    test_df.to_csv(os.path.join(log_dir, f"test_results_{suffix}.csv"), index=False)

    rewards_test = np.array([r['reward'] for r in test_results])
    episodes_test = np.array([r['episode'] for r in test_results])
    r_test_smooth = pd.Series(rewards_test).rolling(smooth_win, min_periods=1).mean()
    r_test_std = pd.Series(rewards_test).rolling(smooth_win, min_periods=1).std().fillna(0)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episodes_test, rewards_test, 'o-', color='steelblue', alpha=0.4, markersize=3, label='原始值')
    plt.fill_between(episodes_test, r_test_smooth - r_test_std, r_test_smooth + r_test_std, color='steelblue', alpha=0.2)
    plt.plot(episodes_test, r_test_smooth, '-', color='darkblue', linewidth=2, label='平滑曲线')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title(f'测试集奖励 (unit{unit_id})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    all_actions = sorted(set(a for r in test_results for a in r['actions'].keys()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(all_actions))))
    for i, action in enumerate(all_actions):
        counts = [r['actions'].get(action, 0) for r in test_results]
        c_smooth = pd.Series(counts).rolling(smooth_win, min_periods=1).mean()
        plt.plot(episodes_test, c_smooth, '-', color=colors[i % 10], linewidth=1.5, label=f'动作{action}')
    plt.xlabel('Episode')
    plt.ylabel('动作次数')
    plt.title('动作分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"test_reward_curves_{suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"unit{unit_id} 测试结果已保存")
