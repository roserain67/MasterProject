import gym
import numpy as np
from gym import spaces
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaintenanceEnv(gym.Env):
    """
    双部件自动运行维护环境。
    每步设备自动运行（pointer 推进），agent 只决定是否维修。
    部件A = HPT效率, 部件B = LPT流量，各自独立退化、独立维修。
    state = [GRU_emb(64), pos_A(1), pos_B(1)] = 66 维
    5 动作: 0=不干预, 1=修A, 2=修B, 3=修AB, 4=更换
    """
    metadata = {"render_modes": []}

    def __init__(self, sequence, encoder, state_dim=66, max_steps=200, step_size=5, cost_alpha=0.3, no_gru=False):
        super().__init__()

        self.sequence = sequence.astype(np.float32)
        self.seq_len = sequence.shape[0]
        self.no_gru = no_gru
        if not no_gru:
            self.encoder = encoder.eval().to(DEVICE)
        else:
            self.encoder = None
        self.max_steps = max_steps

        self.state_dim = state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

        self.R_run = 3.0
        self.C_prev = 3.0
        self.C_prev_AB = 5.0
        self.C_replace = 40.0
        self.penalty_break = 200.0
        self.penalty_over_repair = 100.0
        self.survival_bonus = 50.0

        self.shaping_k = 60.0
        self.gamma = 0.99

        self.step_size = step_size
        self.cost_alpha = cost_alpha

        self.pointer_A = 0
        self.pointer_B = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.max_repair_in_row = 10
        self.current_step = 0
        self.state = None

    def _potential(self):
        return -self.shaping_k * (self.pointer_A + self.pointer_B) / self.seq_len

    def _restore_point(self, count):
        return min((count - 1) * self.step_size, self.seq_len - 1)

    def encode_state(self):
        pos_a = np.array([self.pointer_A / max(1, self.seq_len)], dtype=np.float32)
        pos_b = np.array([self.pointer_B / max(1, self.seq_len)], dtype=np.float32)

        if self.no_gru:
            return np.concatenate([pos_a, pos_b])

        seq_pointer = max(self.pointer_A, self.pointer_B)
        if seq_pointer == 0:
            seq = self.sequence[:1]
        else:
            seq = self.sequence[: seq_pointer + 1]

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.encoder(seq_tensor)
        emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
        return np.concatenate([emb_np, pos_a, pos_b])

    def reset(self):
        self.pointer_A = 0
        self.pointer_B = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.current_step = 0
        self.state = self.encode_state()
        return self.state

    def step(self, action):
        reward = 0.0
        done = False
        self.current_step += 1
        phi_old = self._potential()

        # Phase 1: 维修动作
        if action == 0:
            self.repair_in_row = 0

        elif action == 1:
            self.repair_in_row += 1
            self.repair_count_A += 1
            reward -= self.C_prev * (1 + self.cost_alpha * self.repair_count_A)
            self.pointer_A = min(self.pointer_A, self._restore_point(self.repair_count_A))

        elif action == 2:
            self.repair_in_row += 1
            self.repair_count_B += 1
            reward -= self.C_prev * (1 + self.cost_alpha * self.repair_count_B)
            self.pointer_B = min(self.pointer_B, self._restore_point(self.repair_count_B))

        elif action == 3:
            self.repair_in_row += 1
            self.repair_count_A += 1
            self.repair_count_B += 1
            max_count = max(self.repair_count_A, self.repair_count_B)
            reward -= self.C_prev_AB * (1 + self.cost_alpha * max_count)
            self.pointer_A = min(self.pointer_A, self._restore_point(self.repair_count_A))
            self.pointer_B = min(self.pointer_B, self._restore_point(self.repair_count_B))

        elif action == 4:
            self.repair_in_row += 1
            reward -= self.C_replace
            self.repair_count_A = 0
            self.repair_count_B = 0
            self.pointer_A = 0
            self.pointer_B = 0

        # Phase 2: 设备自动运行
        self.pointer_A += 1
        self.pointer_B += 1

        # Phase 3: 运营收益（无条件）
        reward += self.R_run

        # Phase 3.5: Potential-based reward shaping
        phi_new = self._potential()
        reward += self.gamma * phi_new - phi_old

        # Phase 4: 终止检查
        if self.repair_in_row >= self.max_repair_in_row:
            done = True
            reward -= self.penalty_over_repair

        if self.pointer_A >= self.seq_len - 1 or self.pointer_B >= self.seq_len - 1:
            done = True
            reward -= self.penalty_break

        if not done and self.current_step >= self.max_steps:
            done = True
            reward += self.survival_bonus

        if not done:
            self.state = self.encode_state()

        return self.state, reward, done, {}
