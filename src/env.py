import gym
import numpy as np
from gym import spaces
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaintenanceEnv(gym.Env):
    """
    双部件独立退化维护环境。
    部件A = HPT效率, 部件B = LPT流量，各自独立退化、独立维修。
    state = [GRU_emb(64), pos_A(1), pos_B(1)] = 66 维
    8 动作: 0=运行, 1=修A, 2=修B, 3=修AB, 4=事后修A, 5=事后修B, 6=事后修AB, 7=更换
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
        self.action_space = spaces.Discrete(8)

        self.cost_post_A = 35.0
        self.cost_post_B = 35.0
        self.cost_post_AB = 50.0
        self.cost_replace = 50.0
        self.penalty_break = 200.0
        self.penalty_over_repair = 200.0
        self.survival_bonus = 50.0

        self.step_size = step_size
        self.cost_alpha = cost_alpha

        self.pointer_A = 0
        self.pointer_B = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.max_repair_in_row = 15
        self.current_step = 0
        self.state = None

    def _restore_point(self, count):
        return min((count - 1) * self.step_size, self.seq_len - 1)

    def _scaled_cost(self, base_cost, count):
        return base_cost * (1 + self.cost_alpha * count)

    def _progress_A(self):
        return self.pointer_A / max(1, self.seq_len - 1)

    def _progress_B(self):
        return self.pointer_B / max(1, self.seq_len - 1)

    def _max_progress(self):
        return max(self._progress_A(), self._progress_B())

    def _get_reward_step(self):
        progress = self._max_progress()
        if progress <= 0.5:
            return 3.0
        elif progress <= 0.8:
            return 3.0 - 5.0 * (progress - 0.5) / 0.3
        else:
            return -2.0 - 3.0 * (progress - 0.8) / 0.2

    def _get_preventive_reward(self, progress, count):
        base_reward = 15.0 * progress
        cost_increase = 3.0 * (count - 1)
        return max(0, base_reward - cost_increase)

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
        reward = 0
        done = False
        self.current_step += 1

        step_reward = self._get_reward_step()

        if action == 0:
            self.repair_in_row = 0
            reward += step_reward
            self.pointer_A += 1
            self.pointer_B += 1

        elif action == 1:
            self.repair_in_row += 1
            self.repair_count_A += 1
            reward += self._get_preventive_reward(self._progress_A(), self.repair_count_A)
            reward -= step_reward
            self.pointer_A = self._restore_point(self.repair_count_A)

        elif action == 2:
            self.repair_in_row += 1
            self.repair_count_B += 1
            reward += self._get_preventive_reward(self._progress_B(), self.repair_count_B)
            reward -= step_reward
            self.pointer_B = self._restore_point(self.repair_count_B)

        elif action == 3:
            self.repair_in_row += 1
            self.repair_count_A += 1
            self.repair_count_B += 1
            r_a = self._get_preventive_reward(self._progress_A(), self.repair_count_A)
            r_b = self._get_preventive_reward(self._progress_B(), self.repair_count_B)
            reward += (r_a + r_b) * 0.8
            reward -= step_reward
            self.pointer_A = self._restore_point(self.repair_count_A)
            self.pointer_B = self._restore_point(self.repair_count_B)

        elif action == 4:
            self.repair_in_row += 1
            self.repair_count_A += 1
            reward -= self._scaled_cost(self.cost_post_A, self.repair_count_A)
            reward -= step_reward
            self.pointer_A = self._restore_point(self.repair_count_A)

        elif action == 5:
            self.repair_in_row += 1
            self.repair_count_B += 1
            reward -= self._scaled_cost(self.cost_post_B, self.repair_count_B)
            reward -= step_reward
            self.pointer_B = self._restore_point(self.repair_count_B)

        elif action == 6:
            self.repair_in_row += 1
            self.repair_count_A += 1
            self.repair_count_B += 1
            max_count = max(self.repair_count_A, self.repair_count_B)
            reward -= self._scaled_cost(self.cost_post_AB, max_count)
            reward -= step_reward
            self.pointer_A = self._restore_point(self.repair_count_A)
            self.pointer_B = self._restore_point(self.repair_count_B)

        elif action == 7:
            self.repair_in_row = 0
            reward -= self.cost_replace
            reward -= step_reward
            self.repair_count_A = 0
            self.repair_count_B = 0
            self.pointer_A = 0
            self.pointer_B = 0

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
