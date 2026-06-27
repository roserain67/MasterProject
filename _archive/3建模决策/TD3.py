"""
离散动作 TD3：Twin Q + 延迟策略更新，与 SAC 同环境、同数据，结果保存到指定 log_dir。
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


class GRUEncoder(nn.Module):
    def __init__(self, input_dim=172, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.output_layer = nn.Linear(hidden_dim, 64)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.output_layer(h_n[-1])


def load_gru_encoder():
    model = GRUEncoder().to(DEVICE)
    path = os.path.join(os.path.dirname(__file__), "gru_pretrained.pt")
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, s):
        return F.softmax(self.net(s), dim=-1)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))
        self.net2 = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions))

    def forward(self, s):
        return self.net1(s), self.net2(s)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return map(np.asarray, zip(*batch))

    def __len__(self):
        return len(self.buffer)


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
            for s in np.load(seq_path):
                all_sequences.append(s)
    return all_sequences


def train(encoder_model=None, train_sequences=None, test_by_unit=None, log_dir=None, num_episodes=300):
    if encoder_model is None:
        encoder_model = load_gru_encoder()
    if train_sequences is None:
        data_base = os.path.join(os.path.dirname(__file__), "..", "1数据处理", "DS02", "feature_all")
        train_sequences = load_sequences(data_base, [16, 18, 20])
        if not train_sequences:
            train_sequences = load_sequences(os.path.join(os.path.dirname(__file__), "1数据处理", "DS02", "feature_all"), [16, 18, 20])
        if not train_sequences:
            raise ValueError("未找到训练数据")
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__), "..", "TD3-logs", "1")
    os.makedirs(log_dir, exist_ok=True)

    env = MaintenanceEnv(train_sequences[0], encoder_model, state_dim=64)
    state_dim = 64
    n_actions = env.action_space.n

    actor = Actor(state_dim, n_actions).to(DEVICE)
    critic = TwinCritic(state_dim, n_actions).to(DEVICE)
    critic_t1 = TwinCritic(state_dim, n_actions).to(DEVICE)
    critic_t2 = TwinCritic(state_dim, n_actions).to(DEVICE)
    critic_t1.load_state_dict(critic.state_dict())
    critic_t2.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = optim.Adam(critic.parameters(), lr=3e-4)

    replay = ReplayBuffer()
    MAX_LEN = 200
    GAMMA = 0.99
    BATCH = 64
    TAU = 0.005
    REWARD_CLIP = 200
    TD_CLIP = 800
    GRAD_CLIP = 0.5
    POLICY_DELAY = 2
    INIT_TEMP = 1.2
    TEMP_DECAY_EP = 400
    CURRICULUM_EP = 150

    records = []
    losses_actor = []
    losses_critic = []
    best_reward = -np.inf
    total_updates = 0

    for ep in range(1, num_episodes + 1):
        temp = max(1.0, INIT_TEMP - (INIT_TEMP - 1.0) * max(0, ep - TEMP_DECAY_EP) / max(1, num_episodes - TEMP_DECAY_EP))
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
                    q1_t, q2_t = critic_t1(bs2), critic_t2(bs2)
                    q_next = torch.min(q1_t, q2_t).max(1)[0]
                    td_target = br + GAMMA * (1 - bd) * q_next
                    td_target = torch.clamp(td_target, -TD_CLIP, TD_CLIP)

                q1, q2 = critic(bs)
                q_a1 = q1.gather(1, ba.unsqueeze(1)).squeeze(1)
                q_a2 = q2.gather(1, ba.unsqueeze(1)).squeeze(1)
                loss_c = F.smooth_l1_loss(q_a1, td_target, beta=50.0) + F.smooth_l1_loss(q_a2, td_target, beta=50.0)
                opt_critic.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                opt_critic.step()
                losses_critic.append(loss_c.item())

                total_updates += 1
                if total_updates % POLICY_DELAY == 0:
                    probs_a = actor(bs)
                    logp = torch.log(probs_a.gather(1, ba.unsqueeze(1)).squeeze(1) + 1e-8)
                    q1_a = q1.gather(1, ba.unsqueeze(1)).squeeze(1)
                    advantage = (td_target - q1_a.detach())
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    loss_a = -(logp * advantage).mean()
                    opt_actor.zero_grad()
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), GRAD_CLIP)
                    opt_actor.step()
                    losses_actor.append(loss_a.item())

                for p, pt in zip(critic.parameters(), critic_t1.parameters()):
                    pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)
                for p, pt in zip(critic.parameters(), critic_t2.parameters()):
                    pt.data.copy_(TAU * p.data + (1 - TAU) * pt.data)

            if done:
                break

        mean_entropy = 0.0
        if episode_actions:
            cnt = np.bincount(episode_actions, minlength=n_actions)
            p = cnt / len(episode_actions)
            mean_entropy = -np.sum(p * np.log(p + 1e-8))
        records.append({"episode": ep, "reward": episode_reward, "action_entropy": mean_entropy})

        if episode_reward > best_reward and episode_reward > 50:
            best_reward = episode_reward
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, os.path.join(log_dir, "best_model.pt"))

        if ep % 10 == 0:
            print(f"[TD3] Ep {ep}: reward={episode_reward:.2f}  entropy={mean_entropy:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "loss_reward_action.csv"), index=False)
    L = len(losses_critic)
    pd.DataFrame({"critic": losses_critic, "actor": losses_actor + [0] * (L - len(losses_actor))}).to_csv(os.path.join(log_dir, "loss_curves.csv"), index=False)

    rewards = df["reward"].values
    r_smooth = pd.Series(rewards).rolling(50, min_periods=1).mean()
    r_std = pd.Series(rewards).rolling(50, min_periods=1).std().fillna(0)
    plt.figure()
    plt.plot(np.arange(len(rewards)), rewards, color='steelblue', alpha=0.35)
    plt.fill_between(np.arange(len(rewards)), r_smooth - r_std, r_smooth + r_std, color='steelblue', alpha=0.2)
    plt.plot(np.arange(len(rewards)), r_smooth, color='darkblue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('奖励')
    plt.title('TD3 Episode Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(log_dir, "episode_reward.png"))
    plt.close()

    if test_by_unit:
        _run_test(encoder_model, actor, train_sequences, test_by_unit, log_dir, state_dim, n_actions)

    print(f"TD3 训练完成，结果已保存到 {log_dir}")
    return log_dir


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


if __name__ == "__main__":
    train()
