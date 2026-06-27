import gym
import numpy as np
from gym import spaces
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaintenanceEnv(gym.Env):
    """
    基于 GRU 编码的双部件维护环境。
    state = GRU(序列前缀) ∈ R^64 + 归一化位置 pos_t ∈ [0,1] → R^65
    8 动作: 0=继续运行, 1-3=预防性维修 A/B/AB, 4-6=事后维修 A/B/AB, 7=更换
    """
    metadata = {"render_modes": []}

    def __init__(self, sequence, encoder, state_dim=65, step_size=5, cost_alpha=0.2):
        super().__init__()

        self.sequence = sequence.astype(np.float32)
        self.seq_len = sequence.shape[0]
        self.encoder = encoder.eval().to(DEVICE)

        self.state_dim = state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(8)

        self.reward_step = 1.0
        self.reward_prev_base = 10.0
        self.cost_post_A = 35.0
        self.cost_post_B = 35.0
        self.cost_post_AB = 50.0
        self.cost_replace = 80.0
        self.penalty_break = 100.0
        self.penalty_over_repair = 50.0

        self.step_size = step_size
        self.cost_alpha = cost_alpha

        self.pointer = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.max_repair_in_row = 15
        self.state = None

    def _restore_point(self, count):
        return min((count - 1) * self.step_size, self.seq_len - 1)

    def _scaled_cost(self, base_cost, count):
        return base_cost * (1 + self.cost_alpha * count)

    def encode_state(self):
        if self.pointer == 0:
            seq = self.sequence[:1]
        else:
            seq = self.sequence[: self.pointer + 1]

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = self.encoder(seq_tensor)
        emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)
        pos = np.array([self.pointer / max(1, self.seq_len)], dtype=np.float32)
        return np.concatenate([emb_np, pos])

    def reset(self):
        self.pointer = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.state = self.encode_state()
        return self.state

    def step(self, action):
        reward = 0
        done = False

        if action == 0:
            self.repair_in_row = 0
            reward += self.reward_step
            self.pointer += 1

        elif action in [1, 2, 3]:
            self.repair_in_row += 1
            if action == 1 or action == 3:
                self.repair_count_A += 1
            if action == 2 or action == 3:
                self.repair_count_B += 1
            max_count = max(self.repair_count_A if action in [1, 3] else 0,
                            self.repair_count_B if action in [2, 3] else 0)
            r = self.reward_prev_base - 2.0 * (max_count - 1)
            if action == 3:
                r *= 0.9
            reward += r
            rp_A = self._restore_point(self.repair_count_A) if action in [1, 3] else self.pointer
            rp_B = self._restore_point(self.repair_count_B) if action in [2, 3] else self.pointer
            self.pointer = max(rp_A, rp_B)

        elif action in [4, 5, 6]:
            self.repair_in_row += 1
            if action == 4 or action == 6:
                self.repair_count_A += 1
            if action == 5 or action == 6:
                self.repair_count_B += 1
            max_count = max(self.repair_count_A if action in [4, 6] else 0,
                            self.repair_count_B if action in [5, 6] else 0)
            if action == 4:
                cost = self._scaled_cost(self.cost_post_A, self.repair_count_A)
            elif action == 5:
                cost = self._scaled_cost(self.cost_post_B, self.repair_count_B)
            else:
                cost = self._scaled_cost(self.cost_post_AB, max_count)
            reward -= (cost + self.reward_step)
            rp_A = self._restore_point(self.repair_count_A) if action in [4, 6] else self.pointer
            rp_B = self._restore_point(self.repair_count_B) if action in [5, 6] else self.pointer
            self.pointer = max(rp_A, rp_B)

        elif action == 7:
            self.repair_in_row = 0
            reward -= self.cost_replace
            self.repair_count_A = 0
            self.repair_count_B = 0
            self.pointer = 0

        if self.repair_in_row >= self.max_repair_in_row:
            done = True
            reward -= self.penalty_over_repair

        if self.pointer >= self.seq_len - 1:
            done = True
            if action == 0:
                reward -= self.penalty_break

        if not done:
            self.state = self.encode_state()

        return self.state, reward, done, {}
