"""TD3 基线算法：Twin Q + 延迟策略更新"""
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
from src.encoder import load_gru_encoder
from src.utils.replay_buffer import ReplayBuffer
from src.utils.data_loader import load_sequences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def train(cfg, encoder_model=None, train_sequences=None, test_by_unit=None):
    if encoder_model is None:
        encoder_model = load_gru_encoder(cfg.get("pretrain_path"))
    if train_sequences is None:
        train_sequences, _ = load_sequences(cfg["data_base"], cfg["train_units"])
        if not train_sequences:
            raise ValueError("未找到训练数据")
    if test_by_unit is None:
        test_by_unit = {}
        for uid in cfg.get("test_units", [14]):
            seqs, _ = load_sequences(cfg["data_base"], [uid])
            if seqs:
                test_by_unit[uid] = seqs

    log_dir = cfg["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    state_dim = cfg["state_dim"]
    n_actions = cfg["n_actions"]
    policy_delay = cfg.get("policy_delay", 2)

    actor = Actor(state_dim, n_actions).to(DEVICE)
    critic = TwinCritic(state_dim, n_actions).to(DEVICE)
    critic_target = TwinCritic(state_dim, n_actions).to(DEVICE)
    critic_target.load_state_dict(critic.state_dict())

    opt_actor = optim.Adam(actor.parameters(), lr=cfg["lr_actor"])
    opt_critic = optim.Adam(critic.parameters(), lr=cfg["lr_critic"])
    replay = ReplayBuffer(cfg.get("replay_capacity", 50000))

    num_episodes = cfg["num_episodes"]
    max_len = cfg["max_len"]
    gamma = cfg["gamma"]
    batch_size = cfg["batch_size"]
    tau = cfg["tau"]
    reward_clip = cfg["reward_clip"]
    td_clip = cfg["td_clip"]
    grad_clip = cfg["grad_clip"]
    init_temp = cfg["init_temp"]
    temp_decay_ep = cfg["temp_decay_ep"]
    curriculum_ep = cfg["curriculum_ep"]

    records = []
    losses_actor = []
    losses_critic = []
    best_reward = -np.inf
    total_updates = 0

    for ep in range(1, num_episodes + 1):
        temp = max(1.0, init_temp - (init_temp - 1.0) * max(0, ep - temp_decay_ep) / max(1, num_episodes - temp_decay_ep))
        seq = random.choice(train_sequences)
        env = MaintenanceEnv(seq, encoder_model, state_dim=state_dim)
        s = env.reset()
        episode_reward = 0
        episode_actions = []

        for t in range(max_len):
            s_t = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                logits = actor.net(s_t)
                probs = F.softmax(logits / temp, dim=-1).cpu().numpy().flatten()
            probs = probs / probs.sum()
            if ep <= curriculum_ep:
                probs[4], probs[5], probs[6] = probs[4] * 0.1, probs[5] * 0.1, probs[6] * 0.1
                probs = probs / probs.sum()
            action = np.random.choice(n_actions, p=probs)
            s2, r, done, _ = env.step(action)
            r_clipped = np.clip(r, -reward_clip, reward_clip)
            replay.push(s.copy(), action, r_clipped, s2.copy(), float(done))
            episode_reward += r
            episode_actions.append(action)
            s = s2

            if len(replay) >= batch_size:
                bs, ba, br, bs2, bd = replay.sample(batch_size)
                bs = torch.tensor(bs, dtype=torch.float32, device=DEVICE)
                ba = torch.tensor(ba, dtype=torch.long, device=DEVICE)
                br = torch.tensor(br, dtype=torch.float32, device=DEVICE)
                bs2 = torch.tensor(bs2, dtype=torch.float32, device=DEVICE)
                bd = torch.tensor(bd, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    q1_t, q2_t = critic_target(bs2)
                    q_next = torch.min(q1_t, q2_t).max(1)[0]
                    td_target = br + gamma * (1 - bd) * q_next
                    td_target = torch.clamp(td_target, -td_clip, td_clip)

                q1, q2 = critic(bs)
                q_a1 = q1.gather(1, ba.unsqueeze(1)).squeeze(1)
                q_a2 = q2.gather(1, ba.unsqueeze(1)).squeeze(1)
                loss_c = F.smooth_l1_loss(q_a1, td_target, beta=50.0) + F.smooth_l1_loss(q_a2, td_target, beta=50.0)
                opt_critic.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), grad_clip)
                opt_critic.step()
                losses_critic.append(loss_c.item())

                total_updates += 1
                if total_updates % policy_delay == 0:
                    probs_a = actor(bs)
                    logp = torch.log(probs_a.gather(1, ba.unsqueeze(1)).squeeze(1) + 1e-8)
                    advantage = (td_target - q_a1.detach())
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    loss_a = -(logp * advantage).mean()
                    opt_actor.zero_grad()
                    loss_a.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip)
                    opt_actor.step()
                    losses_actor.append(loss_a.item())

                for p, pt in zip(critic.parameters(), critic_target.parameters()):
                    pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

            if done:
                break

        records.append({"episode": ep, "reward": episode_reward})
        if episode_reward > best_reward and episode_reward > 50:
            best_reward = episode_reward
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, os.path.join(log_dir, "best_model.pt"))
        if ep % 10 == 0:
            print(f"[TD3] Ep {ep}: reward={episode_reward:.2f}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "loss_reward_action.csv"), index=False)
    L = len(losses_critic)
    pd.DataFrame({"critic": losses_critic, "actor": losses_actor + [0] * (L - len(losses_actor))}).to_csv(os.path.join(log_dir, "loss_curves.csv"), index=False)

    if test_by_unit:
        _run_test(encoder_model, actor, test_by_unit, log_dir, state_dim, n_actions, cfg.get("test_episodes", 100))
    print(f"TD3 训练完成，结果已保存到 {log_dir}")
    return log_dir


def _run_test(encoder_model, actor, test_by_unit, log_dir, state_dim, n_actions, num_episodes=100):
    actor.eval()
    for unit_id, test_sequences in test_by_unit.items():
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
            results.append({"episode": ep_idx + 1, "reward": episode_reward,
                            **{f"action_{k}": v for k, v in Counter(episode_actions).items()}})
        pd.DataFrame(results).to_csv(os.path.join(log_dir, f"test_results_unit{unit_id}.csv"), index=False)
    actor.train()
