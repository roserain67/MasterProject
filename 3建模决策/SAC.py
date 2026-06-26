"""
无 PEARL 纯 SAC 稳定版：用于与 PEARL 对照实验。
- 仅 state -> Actor/Critic，无 context、无 z
- 同一维护环境 env_gemini，同一数据与奖励
- 结果保存到 D:\\yyo-Python\\0毕设\\SAC-logs\\1
"""
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

from collections import deque, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env_gemini import MaintenanceEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- GRU（仅用于 state 编码，与 PEARL 一致） ----------
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
        return self.output_layer(h_n[-1])

def load_gru_encoder():
    model = GRUEncoder().to(DEVICE)
    path = os.path.join(os.path.dirname(__file__), "gru_pretrained.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model

# ---------- SAC: Actor / Critic（无 z） ----------
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, s):
        return F.softmax(self.net(s), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, s):
        return self.net(s)

# ---------- 简单 Replay Buffer ----------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, r, s2, d = map(np.asarray, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)

# ---------- 数据加载 ----------
def load_sequences(data_base_path, unit_ids, use_trajectory=True):
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
            seqs = np.load(seq_path)
            for i in range(len(seqs)):
                all_sequences.append(seqs[i])
    return all_sequences

# ---------- 训练 ----------
def _run_test(encoder_model, actor, train_sequences, test_by_unit, log_dir, state_dim, n_actions, num_episodes=100):
    actor.eval()
    for unit_id, test_sequences in test_by_unit.items():
        suffix = f"unit{unit_id}"
        results = []
        for ep_idx in range(num_episodes):
            seq = random.choice(test_sequences)
            env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim)
            s = env.reset()
            episode_reward = 0
            episode_actions = []
            for _ in range(min(200, len(seq))):
                s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    probs = actor(s_t).cpu().numpy().flatten()
                action = np.argmax(probs)
                s2, r, done, _ = env.step(action)
                episode_reward += r
                episode_actions.append(action)
                s = s2
                if done:
                    break
            results.append({"episode": ep_idx + 1, "reward": episode_reward, "actions": dict(Counter(episode_actions))})
        test_df = pd.DataFrame([{"episode": r["episode"], "reward": r["reward"], **{f"action_{k}": v for k, v in r["actions"].items()}} for r in results])
        test_df.to_csv(os.path.join(log_dir, f"test_results_{suffix}.csv"), index=False)
    actor.train()


def train(encoder_model=None, train_sequences=None, test_by_unit=None, log_dir=None, num_episodes=500):
    if encoder_model is None:
        encoder_model = load_gru_encoder()
    if train_sequences is None:
        data_base = os.path.join(os.path.dirname(__file__), "..", "1数据处理", "DS02", "feature_all")
        train_sequences = load_sequences(data_base, [16, 18, 20])
        if len(train_sequences) == 0:
            train_sequences = load_sequences(os.path.join(os.path.dirname(__file__), "1数据处理", "DS02", "feature_all"), [16, 18, 20])
        if len(train_sequences) == 0:
            raise ValueError("未找到训练数据")
    if test_by_unit is None:
        data_base = os.path.join(os.path.dirname(__file__), "..", "1数据处理", "DS02", "feature_all")
        test_seqs_14 = load_sequences(data_base, [14])
        test_by_unit = {14: test_seqs_14 if test_seqs_14 else train_sequences[: min(10, len(train_sequences))]}
    if log_dir is None:
        log_dir = r"D:\yyo-Python\0毕设\SAC-logs\2"
    os.makedirs(log_dir, exist_ok=True)

    env = MaintenanceEnv(train_sequences[0], encoder_model, state_dim=64)
    state_dim = 64
    n_actions = env.action_space.n

    actor = Actor(state_dim, n_actions).to(DEVICE)
    critic = Critic(state_dim, n_actions).to(DEVICE)
    critic_target = Critic(state_dim, n_actions).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)

    replay = ReplayBuffer()
    LOG_DIR = log_dir

    NUM_EPISODES = num_episodes
    MAX_LEN = 200
    GAMMA = 0.99
    BATCH = 64
    TAU = 0.005
    REWARD_CLIP = 200
    TD_CLIP = 800
    GRAD_CLIP = 0.5
    ENTROPY_COEF = 0.05
    INIT_TEMP = 1.2
    TEMP_DECAY_EP = 400
    CURRICULUM_EP = 150

    records = []
    losses_actor = []
    losses_critic = []
    best_reward = -np.inf

    for ep in range(1, NUM_EPISODES + 1):
        temp = max(1.0, INIT_TEMP - (INIT_TEMP - 1.0) * max(0, ep - TEMP_DECAY_EP) / max(1, NUM_EPISODES - TEMP_DECAY_EP))
        seq = random.choice(train_sequences)
        env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim)
        s = env.reset()
        episode_reward = 0
        episode_actions = []

        for t in range(MAX_LEN):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = actor.net(s_t)
                probs = F.softmax(logits / temp, dim=-1).cpu().numpy().flatten()
            probs = probs / probs.sum()
            if ep <= CURRICULUM_EP:
                probs[4], probs[5], probs[6] = probs[4] * 0.1, probs[5] * 0.1, probs[6] * 0.1
                probs = probs / probs.sum()
            action = np.random.choice(n_actions, p=probs)
            s2, r, done, _ = env.step(action)
            r_clipped = np.clip(r, -REWARD_CLIP, REWARD_CLIP)
            replay.push(s.copy(), action, r_clipped, s2.copy(), float(done))
            episode_reward += r
            episode_actions.append(action)
            s = s2

            if len(replay) >= BATCH:
                bs, ba, br, bs2, bd = replay.sample(BATCH)
                bs = torch.tensor(bs, dtype=torch.float32, device=DEVICE)
                ba = torch.tensor(ba, dtype=torch.long, device=DEVICE)
                br = torch.tensor(br, dtype=torch.float32, device=DEVICE)
                bs2 = torch.tensor(bs2, dtype=torch.float32, device=DEVICE)
                bd = torch.tensor(bd, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    q_next = critic_target(bs2).max(1)[0]
                    td_target = br + GAMMA * (1 - bd) * q_next
                    td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                q = critic(bs)
                q_a = q.gather(1, ba.unsqueeze(1)).squeeze(1)
                loss_c = F.smooth_l1_loss(q_a, td_target, beta=50.0)
                opt_critic.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                opt_critic.step()

                probs_a = actor(bs)
                logp = torch.log(probs_a.gather(1, ba.unsqueeze(1)).squeeze(1) + 1e-8)
                with torch.no_grad():
                    advantage = (td_target - q_a.detach())
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                loss_a = -(logp * advantage).mean()
                entropy = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1).mean()
                loss_a = loss_a - ENTROPY_COEF * entropy
                opt_actor.zero_grad()
                loss_a.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                opt_actor.step()

                for p, pt in zip(critic.parameters(), critic_target.parameters()):
                    pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

                losses_actor.append(loss_a.item())
                losses_critic.append(loss_c.item())

            if done:
                break

        mean_entropy = -np.sum(np.bincount(episode_actions, minlength=n_actions) / len(episode_actions) * np.log(np.bincount(episode_actions, minlength=n_actions) / len(episode_actions) + 1e-8)) if episode_actions else 0.0
        records.append({
            "episode": ep,
            "reward": episode_reward,
            "action_entropy": mean_entropy,
            "actions": Counter(episode_actions)
        })

        if episode_reward > best_reward and episode_reward > 50:
            best_reward = episode_reward
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, os.path.join(LOG_DIR, "best_model.pt"))

        if ep % 10 == 0:
            print(f"Ep {ep}: reward={episode_reward:.2f}  action_entropy={mean_entropy:.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "actions"} for r in records])
    df.to_csv(os.path.join(LOG_DIR, "loss_reward_action.csv"), index=False)
    pd.DataFrame({"critic": losses_critic, "actor": losses_actor}).to_csv(os.path.join(LOG_DIR, "loss_curves.csv"), index=False)

    rewards = df["reward"].values
    r_smooth = pd.Series(rewards).rolling(50, min_periods=1).mean()
    r_std = pd.Series(rewards).rolling(50, min_periods=1).std().fillna(0)
    plt.figure()
    plt.plot(np.arange(len(rewards)), rewards, color='steelblue', alpha=0.35, linewidth=0.8)
    plt.fill_between(np.arange(len(rewards)), r_smooth - r_std, r_smooth + r_std, color='steelblue', alpha=0.2)
    plt.plot(np.arange(len(rewards)), r_smooth, color='darkblue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('SAC Episode Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(LOG_DIR, "episode_reward.png"))
    plt.close()

    lc = pd.Series(losses_critic)
    la = pd.Series(losses_actor)
    x = np.arange(len(losses_critic))
    plt.figure()
    plt.plot(x, lc.rolling(50, min_periods=1).mean(), color='darkred', label='critic')
    plt.plot(x, la.rolling(50, min_periods=1).mean(), color='darkgreen', label='actor')
    plt.legend()
    plt.xlabel('更新步数')
    plt.ylabel('Loss')
    plt.title('SAC Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(LOG_DIR, "loss_curves.png"))
    plt.close()

    if test_by_unit:
        _run_test(encoder_model, actor, train_sequences, test_by_unit, LOG_DIR, state_dim, n_actions)

    print(f"SAC 训练完成，结果已保存到 {LOG_DIR}")
    return LOG_DIR


if __name__ == "__main__":
    train()
