'''
如果你后续想把 GRU 移出 env、放到 agent 内并让 actor 更新 GRU（真正端到端），我可以帮你做更改，
那需要把 env 返回原始序列（或序列切片）并在 agent 中用 GRU forward，
然后按 state.detach() / state 的策略进行 critic/actor 更新。
QQQ问题在于，现在的不是端到端？
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

from collections import deque, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env_gemini import MaintenanceEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 加载你的真实 GRUEncoder
# ======================================================
class GRUEncoder(nn.Module):
    def __init__(self, input_dim=172, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.output_layer = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        output, h_n = self.gru(x)
        emb = self.output_layer(h_n[-1])
        return emb  # (batch, 64)

def load_real_gru_encoder():
    model = GRUEncoder().to(DEVICE)
    pretrain_path = os.path.join(os.path.dirname(__file__), "gru_pretrained.pt")
    if os.path.exists(pretrain_path):
        model.load_state_dict(torch.load(pretrain_path, map_location=DEVICE))
        print(">> GRU 模型已加载（预训练权重）")
    else:
        print(">> GRU 模型已加载（随机初始化）。运行 pretrain_gru.py 可预训练健康编码")
    return model
# 2. Context Encoder (PEARL 任务推断部分)
# ======================================================
Z_DIM = 8
CONTEXT_K = 16

class ContextEncoder(nn.Module):
    def __init__(self, input_dim, hidden=128, z_dim=Z_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, z_dim)
        self.fc_logvar = nn.Linear(hidden, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 5)  # 防止数值爆炸
        return mu, logvar

def sample_z(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std

# Actor / Critic
# ======================================================
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

# Replay Buffer + Context Buffer
# ======================================================
class EpisodeReplayBuffer:
    """按 episode 存储，支持任务平滑采样（Batch 同时包含 Unit 16/18/20）"""
    TARGET_UNITS = [16, 18, 20]

    def __init__(self, good_capacity=80, recent_capacity=40):
        self.good_by_unit = {u: deque(maxlen=good_capacity) for u in self.TARGET_UNITS}
        self.recent_by_unit = {u: deque(maxlen=recent_capacity) for u in self.TARGET_UNITS}

    def push_episode(self, transitions, episode_reward, unit_id, good_threshold=500, min_threshold=0):
        if len(transitions) < 2 or unit_id not in self.TARGET_UNITS:
            return
        item = (list(transitions), episode_reward)
        if episode_reward >= good_threshold:
            self.good_by_unit[unit_id].append(item)
        if episode_reward >= min_threshold:
            self.recent_by_unit[unit_id].append(item)

    def _context_from_transitions(self, transitions, k=CONTEXT_K):
        k = min(k, len(transitions))
        batch = random.sample(transitions, k)
        vecs = [np.concatenate([s, [a, r], s2]) for (s, a, r, s2, d) in batch]
        return np.mean(vecs, axis=0).astype(np.float32)

    def sample(self, batch_size):
        use_good = any(len(self.good_by_unit[u]) > 0 for u in self.TARGET_UNITS) and random.random() < 0.7
        pools = {u: (self.good_by_unit[u] if use_good and len(self.good_by_unit[u]) > 0 else self.recent_by_unit[u])
                 for u in self.TARGET_UNITS}
        for u in self.TARGET_UNITS:
            if len(pools[u]) == 0:
                pools[u] = self.good_by_unit[u] if len(self.good_by_unit[u]) > 0 else self.recent_by_unit[u]

        # 任务平滑：若三个 Unit 都有数据，各采 batch_size//3，梯度取三任务交集
        available_units = [u for u in self.TARGET_UNITS if len(pools[u]) > 0]
        if len(available_units) == 0:
            return None

        if len(available_units) >= 3:
            n_per = batch_size // 3
            s_list, a_list, r_list, s2_list, d_list, ctx_list = [], [], [], [], [], []
            for u in self.TARGET_UNITS:
                if len(pools[u]) == 0:
                    continue
                ep_transitions, _ = random.choice(pools[u])
                if len(ep_transitions) < 2:
                    continue
                n = min(n_per, len(ep_transitions))
                indices = random.choices(range(len(ep_transitions)), k=n)
                batch = [ep_transitions[i] for i in indices]
                s, a, r, s2, d = map(np.asarray, zip(*batch))
                ctx = self._context_from_transitions(ep_transitions)
                s_list.append(s); a_list.append(a); r_list.append(r); s2_list.append(s2); d_list.append(d)
                ctx_list.append((ctx, n))
            if len(ctx_list) < 2:
                available_units = available_units[:1]
            else:
                s_all = np.concatenate(s_list, axis=0)
                a_all = np.concatenate(a_list, axis=0)
                r_all = np.concatenate(r_list, axis=0)
                s2_all = np.concatenate(s2_list, axis=0)
                d_all = np.concatenate(d_list, axis=0)
                return s_all, a_all, r_all, s2_all, d_all, ctx_list

        # 回退：单任务采样
        u = random.choice(available_units)
        ep_transitions, _ = random.choice(pools[u])
        if len(ep_transitions) < 2:
            return None
        n = min(batch_size, len(ep_transitions))
        indices = random.choices(range(len(ep_transitions)), k=n)
        batch = [ep_transitions[i] for i in indices]
        s, a, r, s2, d = map(np.asarray, zip(*batch))
        ctx = self._context_from_transitions(ep_transitions)
        return s, a, r, s2, d, [(ctx, n)]

    def __len__(self):
        return sum(len(self.good_by_unit[u]) + len(self.recent_by_unit[u]) for u in self.TARGET_UNITS)

class ContextBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def clear(self):
        self.buffer.clear()

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample_context(self, k=CONTEXT_K):
        k = min(k, len(self.buffer))
        if k == 0:
            return None
        batch = random.sample(self.buffer, k)
        vecs = []
        for (s, a, r, s2, d) in batch:
            vec = np.concatenate([s, [a, r], s2])
            vecs.append(vec)
        return np.mean(vecs, axis=0).astype(np.float32)

# 3. 数据加载
# ======================================================
def load_sequences(data_base_path, unit_ids, use_trajectory=True):
    """
    加载序列数据，返回 (sequences, unit_ids) 便于任务平滑采样。
    优先使用 trajectory_complete.npy，否则 sequences_complete.npy。
    """
    all_sequences = []
    all_unit_ids = []
    for unit_id in unit_ids:
        traj_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "trajectory_complete.npy")
        seq_path = os.path.join(data_base_path, f"unit{unit_id}", "feature_selected", "sequences_complete.npy")
        if use_trajectory and os.path.exists(traj_path):
            traj = np.load(traj_path)
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

# 4. 训练主程序
# ======================================================
def train():
    # ---------- 加载序列 ----------
    data_base = os.path.join("..", "1数据处理", "DS02", "feature_all")
    train_sequences, train_unit_ids = load_sequences(data_base, [16, 18, 20])
    if len(train_sequences) == 0:
        train_sequences, train_unit_ids = load_sequences(os.path.join("1数据处理", "DS02", "feature_all"), [16, 18, 20])
    if len(train_sequences) == 0:
        raise ValueError("未找到训练数据，请检查路径")

    # 测试集：分别加载 unit14、unit15，测试与保存时按 unit 区分
    test_units = [14, 15]
    test_by_unit = {}
    for uid in test_units:
        seqs, _ = load_sequences(data_base, [uid])
        if len(seqs) > 0:
            test_by_unit[uid] = seqs
        else:
            seqs_alt, _ = load_sequences(os.path.join("1数据处理", "DS02", "feature_all"), [uid])
            if len(seqs_alt) > 0:
                test_by_unit[uid] = seqs_alt
    if not test_by_unit:
        test_by_unit[14] = train_sequences[: min(10, len(train_sequences))]  # fallback

    encoder_model = load_real_gru_encoder()
    env = MaintenanceEnv(train_sequences[0], encoder_model, state_dim=64)
    state_dim = 64  # 仅 GRU(64)，无 degradation
    n_actions = env.action_space.n

    # ---------- PEARL modules ----------
    context_input_dim = state_dim + 2 + state_dim
    encoder = ContextEncoder(context_input_dim).to(DEVICE)
    actor = Actor(state_dim, Z_DIM, n_actions).to(DEVICE)
    critic = Critic(state_dim, Z_DIM, n_actions).to(DEVICE)

    opt_enc = optim.Adam(encoder.parameters(), lr=4e-4)  # 提高 encoder LR，多任务下 z 需快速区分
    opt_actor = optim.Adam(actor.parameters(), lr=2e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=5e-5)  # 更低 LR 稳定 Critic，避免 Q 爆炸

    replay = EpisodeReplayBuffer()
    ctxbuf = ContextBuffer()

    WARMUP_EP = 25           # 前 25 ep 全部加入 replay，保证冷启动有数据
    REPLAY_GOOD_THRESH = 120 # 可达（MAX_LEN=200, reward_step=1 下约 120 步即 120）
    REPLAY_MIN_THRESH = -3000 # 负回报也加入 recent，否则后期几乎无新经验、replay 枯竭
    COLLAPSE_THRESH = 80     # 连续低于此 reward 则触发恢复
    COLLAPSE_WINDOW = 8      # 连续 8 ep 低于阈值则恢复

    # ---------- Logging ----------
    LOG_DIR = r"D:\yyo-Python\0毕设\logs\13"
    os.makedirs(LOG_DIR, exist_ok=True)

    # ---------- Critic 稳定性 ----------
    critic_target = Critic(state_dim, Z_DIM, n_actions).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())
    TAU = 0.005   # 软更新系数（降低以吸收多任务切换的梯度冲击）
    REWARD_CLIP = 200   # 单步奖励裁剪
    TD_CLIP = 800       # TD 目标裁剪（允许正 return，但抑制极端值）
    GRAD_CLIP = 0.5    # 梯度裁剪（收紧）

    records = []
    losses_actor = []
    losses_critic = []

    # ---------- Training Loop ----------
    NUM_EPISODES = 600
    MAX_LEN = 200
    GAMMA = 0.99
    BATCH = 64
    ENTROPY_COEF = 0.05  # 熵正则，防止策略塌缩（增大以保持探索）
    INIT_TEMP = 1.4      # 初始探索温度
    TEMP_DECAY_EP = 300  # 温度在 ep 300 后才衰减，延长探索期

    total_steps = 0
    best_reward = -np.inf

    CURRICULUM_EP = 150  # 前 150 ep 压制事后维修(4/5/6)
    low_reward_streak = 0  # 连续低 reward 计数，用于塌缩恢复

    for ep in range(1, NUM_EPISODES + 1):
        ctxbuf.clear()  # 每 episode 重置 context，防止塌缩经验污染 z
        temp = max(1.0, INIT_TEMP - (INIT_TEMP - 1.0) * max(0, ep - TEMP_DECAY_EP) / max(1, NUM_EPISODES - TEMP_DECAY_EP))
        idx = random.randint(0, len(train_sequences) - 1)
        seq = train_sequences[idx]
        unit_id = train_unit_ids[idx]
        env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim)
        s = env.reset()
        episode_reward = 0
        episode_actions = []
        ep_transitions = []  # 本 episode 的转移，结束时按 reward 决定是否加入 replay
        ep_entropies = []    # 本幕动作熵（用于打印）
        ep_z_norms = []     # 本幕 z 向量 L2 范数（用于打印）

        for t in range(MAX_LEN):
            total_steps += 1
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # ---------- 计算 z ----------
            ctx = ctxbuf.sample_context()
            if ctx is None:
                ctx_t = torch.zeros((1, context_input_dim), device=DEVICE)
            else:
                ctx_t = torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            mu, logvar = encoder(ctx_t)
            z = sample_z(mu, logvar)

            # ---------- Actor 选择动作（带温度探索） ----------
            with torch.no_grad():
                logits = actor.net(torch.cat([s_t, z], dim=-1))
                probs = F.softmax(logits / temp, dim=-1).cpu().numpy().flatten()
            probs = probs / probs.sum()
            # Curriculum：前 CURRICULUM_EP 个 episode 压制事后维修(4/5/6)，降低塌缩风险
            if ep <= CURRICULUM_EP:
                probs[4], probs[5], probs[6] = probs[4] * 0.1, probs[5] * 0.1, probs[6] * 0.1
                probs = probs / probs.sum()

            action = np.random.choice(n_actions, p=probs)
            entropy_step = -np.sum(probs * np.log(probs + 1e-8))
            z_norm_step = float(torch.norm(z).item())
            ep_entropies.append(entropy_step)
            ep_z_norms.append(z_norm_step)

            s2, r, done, _ = env.step(action)

            r_clipped = np.clip(r, -REWARD_CLIP, REWARD_CLIP)
            ep_transitions.append((s.copy(), action, r_clipped, s2.copy(), done))
            ctxbuf.push(s, action, r_clipped, s2, done)

            episode_reward += r
            episode_actions.append(action)

            s = s2

            # ---------- 更新（Critic 每 2 步更新一次，平衡 Actor/Critic） ----------
            sample_result = replay.sample(BATCH) if len(replay) > 0 else None
            if sample_result is not None:
                bs, ba, br, bs2, bd, ctx_list = sample_result
                n_batch = len(bs)
                if n_batch >= 32:  # 至少 32 条才更新
                    bs = torch.tensor(bs, dtype=torch.float32, device=DEVICE)
                    ba = torch.tensor(ba, dtype=torch.long, device=DEVICE)
                    br = torch.tensor(br, dtype=torch.float32, device=DEVICE)
                    bs2 = torch.tensor(bs2, dtype=torch.float32, device=DEVICE)
                    bd = torch.tensor(bd, dtype=torch.float32, device=DEVICE)

                    # 任务平滑：每个 (ctx, n) 对应一个 Unit，分别推断 z 后拼接
                    z_parts, mu_list, logvar_list = [], [], []
                    for ctx_vec, n in ctx_list:
                        ctx_b = torch.tensor(ctx_vec, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        mu, logvar = encoder(ctx_b)
                        z = sample_z(mu, logvar).repeat(n, 1)
                        z_parts.append(z)
                        mu_list.append(mu)
                        logvar_list.append(logvar)
                    z_b = torch.cat(z_parts, dim=0)
                    mu_b = torch.cat(mu_list, dim=0)
                    logvar_b = torch.cat(logvar_list, dim=0)

                    # Critic 使用 detached z（避免更新 encoder）
                    z_b_for_critic = z_b.detach()

                    # Critic: Huber loss + 奖励裁剪，抑制大误差
                    q = critic(bs, z_b_for_critic)
                    q_a = q.gather(1, ba.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        q_next = critic_target(bs2, z_b_for_critic).max(1)[0]
                        br_clipped = torch.clamp(br, -REWARD_CLIP, REWARD_CLIP)
                        td_target = br_clipped + GAMMA * (1 - bd) * q_next
                        td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                    loss_c = F.smooth_l1_loss(q_a, td_target, beta=50.0)  # Huber 替代 MSE
                    probs = actor(bs, z_b)
                    logp = torch.log(probs.gather(1, ba.unsqueeze(1)).squeeze(1) + 1e-8)
                    advantage = (td_target - q_a).detach()
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    loss_a = -(logp * advantage).mean()
                    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                    loss_a = loss_a - ENTROPY_COEF * entropy
                    # KL 散度正则，强迫 encoder 把不同 Unit 的特征拉开
                    kl = -0.5 * (1 + logvar_b - mu_b.pow(2) - logvar_b.exp()).mean()
                    reg = 0.1 * (mu_b.pow(2).mean() + logvar_b.exp().mean()) + 0.2 * kl

                    opt_critic.zero_grad()
                    opt_actor.zero_grad()
                    opt_enc.zero_grad()

                    update_critic = (total_steps % 2 == 0)
                    total_loss = (loss_c if update_critic else loss_c.detach()) + loss_a + reg
                    total_loss.backward()

                    if update_critic:
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                        opt_critic.step()
                        for p, pt in zip(critic.parameters(), critic_target.parameters()):
                            pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
                    torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(encoder.parameters()), GRAD_CLIP)
                    opt_actor.step()
                    opt_enc.step()

                    losses_actor.append(loss_a.item())
                    losses_critic.append(loss_c.item())

            if done:
                break

        # 按 episode 加入 replay：warmup 全加，之后只加 reward 达标的，好经验进 good buffer
        min_th = -9999 if ep <= WARMUP_EP else REPLAY_MIN_THRESH
        replay.push_episode(ep_transitions, episode_reward, unit_id, REPLAY_GOOD_THRESH, min_th)

        # 塌缩检测与恢复
        if episode_reward < COLLAPSE_THRESH:
            low_reward_streak += 1
        else:
            low_reward_streak = 0
        best_path = os.path.join(LOG_DIR, "best_model.pt")
        if low_reward_streak >= COLLAPSE_WINDOW and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE)
            actor.load_state_dict(ckpt["actor"])
            encoder.load_state_dict(ckpt["encoder"])
            critic.load_state_dict(ckpt["critic"])
            critic_target.load_state_dict(critic.state_dict())
            for g in opt_actor.param_groups:
                g["lr"] *= 0.5
            for g in opt_enc.param_groups:
                g["lr"] *= 0.5
            low_reward_streak = 0
            print(f"Ep {ep}: 塌缩恢复，已加载 best_model，LR 减半")

        mean_entropy = float(np.mean(ep_entropies)) if ep_entropies else 0.0
        mean_z_norm = float(np.mean(ep_z_norms)) if ep_z_norms else 0.0
        records.append({
            "episode": ep,
            "reward": episode_reward,
            "action_entropy": mean_entropy,
            "z_norm": mean_z_norm,
            "actions": Counter(episode_actions)
        })

        if episode_reward > best_reward and episode_reward > 50:
            best_reward = episode_reward
            torch.save({
                "actor": actor.state_dict(),
                "encoder": encoder.state_dict(),
                "critic": critic.state_dict(),
            }, os.path.join(LOG_DIR, "best_model.pt"))
            print(f"Ep {ep}: reward={episode_reward:.2f} [best saved]")

        if ep % 10 == 0:
            print(f"Ep {ep}: reward={episode_reward:.2f}  action_entropy={mean_entropy:.4f}  z_norm={mean_z_norm:.4f}")

    # ---------- 保存 CSV ----------
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(LOG_DIR, "loss_reward_action.csv"), index=False)

    df_loss = pd.DataFrame({
        "critic": losses_critic,
        "actor": losses_actor
    })
    df_loss.to_csv(os.path.join(LOG_DIR, "loss_curves.csv"), index=False)

    # ---------- 绘图（平滑 + 方差范围 mean±std） ----------
    SMOOTH_WIN = 50  # 平滑窗口

    # Episode Reward：浅色真实值 + 方差范围(mean±std)，深色平滑曲线
    rewards = df["reward"].values
    r_smooth = pd.Series(rewards).rolling(SMOOTH_WIN, min_periods=1).mean()
    r_std = pd.Series(rewards).rolling(SMOOTH_WIN, min_periods=1).std().fillna(0)
    x = np.arange(len(rewards))
    plt.plot(x, rewards, color='steelblue', alpha=0.35, linewidth=0.8, label='原始值')
    plt.fill_between(x, r_smooth - r_std, r_smooth + r_std, color='steelblue', alpha=0.2)
    plt.plot(x, r_smooth, color='darkblue', linewidth=2, label='平滑曲线')
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('Episode Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(LOG_DIR, "episode_reward.png"))
    plt.close()

    # Loss Curves：浅色原始 + 方差范围(mean±std)，深色平滑
    lc = pd.Series(losses_critic)
    la = pd.Series(losses_actor)
    lc_smooth = lc.rolling(SMOOTH_WIN, min_periods=1).mean()
    la_smooth = la.rolling(SMOOTH_WIN, min_periods=1).mean()
    lc_std = lc.rolling(SMOOTH_WIN, min_periods=1).std().fillna(0)
    la_std = la.rolling(SMOOTH_WIN, min_periods=1).std().fillna(0)
    x_loss = np.arange(len(losses_critic))
    plt.plot(x_loss, lc, color='coral', alpha=0.3, linewidth=0.6)
    plt.fill_between(x_loss, lc_smooth - lc_std, lc_smooth + lc_std, color='coral', alpha=0.2)
    plt.plot(x_loss, lc_smooth, color='darkred', linewidth=1.5, label='critic')
    plt.plot(x_loss, la, color='seagreen', alpha=0.3, linewidth=0.6)
    plt.fill_between(x_loss, la_smooth - la_std, la_smooth + la_std, color='seagreen', alpha=0.2)
    plt.plot(x_loss, la_smooth, color='darkgreen', linewidth=1.5, label='actor')
    plt.legend()
    plt.xlabel('更新步数')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(LOG_DIR, "loss_curves.png"))
    plt.close()

    print(f"训练完成！日志与图像已保存到 {LOG_DIR}")

    # ---------- 测试评估：按 unit14、unit15 分别测试并保存 ----------
    SMOOTH_TEST = 10
    for unit_id, test_sequences in test_by_unit.items():
        test_results = evaluate(encoder_model, actor, encoder, test_sequences, state_dim, n_actions, context_input_dim, num_episodes=300)
        suffix = f"unit{unit_id}"

        test_df = pd.DataFrame([{
            'episode': r['episode'],
            'reward': r['reward'],
            **{f'action_{k}': v for k, v in r['actions'].items()}
        } for r in test_results])
        test_df.to_csv(os.path.join(LOG_DIR, f"test_results_{suffix}.csv"), index=False)

        rewards_test = np.array([r['reward'] for r in test_results])
        episodes_test = np.array([r['episode'] for r in test_results])
        r_test_smooth = pd.Series(rewards_test).rolling(SMOOTH_TEST, min_periods=1).mean()
        r_test_std = pd.Series(rewards_test).rolling(SMOOTH_TEST, min_periods=1).std().fillna(0)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(episodes_test, rewards_test, 'o-', color='steelblue', alpha=0.4, markersize=3, label='原始值')
        plt.fill_between(episodes_test, r_test_smooth - r_test_std, r_test_smooth + r_test_std, color='steelblue', alpha=0.2)
        plt.plot(episodes_test, r_test_smooth, '-', color='darkblue', linewidth=2, label='平滑曲线')
        plt.xlabel('Episode')
        plt.ylabel('奖励')
        plt.title(f'测试集奖励曲线 (unit{unit_id})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        all_actions = sorted(set(a for r in test_results for a in r['actions'].keys()))
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(all_actions))))
        for i, action in enumerate(all_actions):
            counts = [r['actions'].get(action, 0) for r in test_results]
            c_smooth = pd.Series(counts).rolling(SMOOTH_TEST, min_periods=1).mean()
            c = colors[i % 10]
            plt.plot(episodes_test, counts, 'o-', color=c, alpha=0.35, markersize=2)
            plt.plot(episodes_test, c_smooth, '-', color=c, linewidth=1.5, label=f'动作{action}')
        plt.xlabel('Episode')
        plt.ylabel('动作次数')
        plt.title('动作分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(LOG_DIR, f"test_reward_curves_{suffix}.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"unit{unit_id} 测试结果已保存: test_results_{suffix}.csv, test_reward_curves_{suffix}.png")
    print(f"全部测试结果已保存到 {LOG_DIR}")

# 5. 测试评估
# ======================================================
def evaluate(encoder_model, actor, context_encoder, test_sequences, state_dim, n_actions, context_input_dim, num_episodes=10, deterministic=True):
    """
    在测试集上评估模型。
    deterministic: 使用 argmax 替代随机采样，评估更稳定。
    测试前预填充 context，减小 z 分布偏移。
    """
    actor.eval()
    context_encoder.eval()

    test_results = []
    ctxbuf = ContextBuffer()

    # 预填充 context：用训练序列做若干步 rollout，使 context buffer 非空
    prefill_seqs = test_sequences if len(test_sequences) > 0 else []
    for _ in range(min(3, len(prefill_seqs) or 1)):
        if prefill_seqs:
            seq = random.choice(prefill_seqs)
        else:
            break
        env_pre = MaintenanceEnv(seq, encoder_model, state_dim=state_dim)
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
        env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim)
        s = env.reset()
        episode_reward = 0
        episode_actions = []
        MAX_LEN = min(200, len(seq))

        for t in range(MAX_LEN):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            ctx = ctxbuf.sample_context()
            if ctx is None:
                ctx_t = torch.zeros((1, context_input_dim), device=DEVICE)
            else:
                ctx_t = torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)

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



if __name__ == "__main__":
    train()
