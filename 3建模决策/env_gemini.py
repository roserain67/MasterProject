import gym
import numpy as np
from gym import spaces
import torch

class MaintenanceEnv(gym.Env):
    """
    基于原始参数信号（GRU 编码）的维护环境，不含 degradation/RUL 类输入。
    - state 仅由 GRU(序列) 构成，无轨迹位置标量
    - 奖励仅基于动作类型与是否到达轨迹末尾等结果
    """
    metadata = {"render_modes": []}

    def __init__(self, sequence, encoder, state_dim=64, step_size=5, cost_alpha=0.2):
        super().__init__()

        self.sequence = sequence.astype(np.float32)
        self.seq_len = sequence.shape[0]
        self.encoder = encoder.eval().cuda()

        self.state_dim = state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # 8 个动作：0继续运行，1-3提前维修A/B/AB，4-6事后维修A/B/AB，7更换
        self.action_space = spaces.Discrete(8)

        self.reward_step = 1.0
        self.reward_prev_base = 10.0     # 提前维修固定小额正分，不依赖 degradation
        self.cost_post_A = 35.0
        self.cost_post_B = 35.0
        self.cost_post_AB = 50.0
        self.cost_replace = 80.0
        self.penalty_break = 100.0       # 运行到序列末尾（设备彻底损坏）
        self.penalty_over_repair = 50.0  # 过度维修（死循环）

        self.step_size = step_size
        self.cost_alpha = cost_alpha

        # 内部指针与维修计数
        self.pointer = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.max_repair_in_row = 15
        self.state = None

    def _restore_point(self, count):
        """第 count 次维修后的恢复点：第1次=0，第n次=(n-1)*step_size"""
        return min((count - 1) * self.step_size, self.seq_len - 1)

    def _scaled_cost(self, base_cost, count):
        """成本递增：第n次维修成本 = base * (1 + alpha * n)"""
        return base_cost * (1 + self.cost_alpha * count)

    def encode_state(self):
        """序列截断 -> GRU -> 仅返回编码向量，不拼接 degradation/位置"""
        if self.pointer == 0:
            seq = self.sequence[:1]
        else:
            seq = self.sequence[: self.pointer + 1]

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            emb = self.encoder(seq_tensor)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    # ---------------- reset ---------------- #
    def reset(self):
        self.pointer = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0
        self.state = self.encode_state()
        return self.state

    # ---------------- step ---------------- #
    def step(self, action):
        reward = 0
        done = False

        # 0：继续运行
        if action == 0:
            self.repair_in_row = 0
            reward += self.reward_step
            self.pointer += 1

        # 1-3：提前预防性维修，奖励仅与动作与次数相关，不依赖 degradation
        elif action in [1, 2, 3]:
            self.repair_in_row += 1
            if action == 1 or action == 3:
                self.repair_count_A += 1
            if action == 2 or action == 3:
                self.repair_count_B += 1
            max_count = max(self.repair_count_A if action in [1,3] else 0,
                            self.repair_count_B if action in [2,3] else 0)
            r = self.reward_prev_base - 2.0 * (max_count - 1)
            if action == 3:
                r *= 0.9
            reward += r
            rp_A = self._restore_point(self.repair_count_A) if action in [1,3] else self.pointer
            rp_B = self._restore_point(self.repair_count_B) if action in [2,3] else self.pointer
            self.pointer = max(rp_A, rp_B)

        # 4-6：事后纠正性维修
        elif action in [4, 5, 6]:
            self.repair_in_row += 1
            if action == 4 or action == 6:
                self.repair_count_A += 1
            if action == 5 or action == 6:
                self.repair_count_B += 1
            max_count = max(self.repair_count_A if action in [4,6] else 0,
                            self.repair_count_B if action in [5,6] else 0)
            if action == 4:
                cost = self._scaled_cost(self.cost_post_A, self.repair_count_A)
            elif action == 5:
                cost = self._scaled_cost(self.cost_post_B, self.repair_count_B)
            else:
                cost = self._scaled_cost(self.cost_post_AB, max_count)
            reward -= (cost + self.reward_step)
            rp_A = self._restore_point(self.repair_count_A) if action in [4,6] else self.pointer
            rp_B = self._restore_point(self.repair_count_B) if action in [5,6] else self.pointer
            self.pointer = max(rp_A, rp_B)

        # 7：更换
        elif action == 7:
            self.repair_in_row = 0
            reward -= self.cost_replace
            self.repair_count_A = 0
            self.repair_count_B = 0
            self.pointer = 0

        # ---------------- 边界与惩罚检查 ----------------
        # 1. 过度维修早停
        if self.repair_in_row >= self.max_repair_in_row:
            done = True
            reward -= self.penalty_over_repair

        # 2. [关键修改] 到达序列末尾（意味着设备彻底损坏且未被更换）
        if self.pointer >= self.seq_len - 1:
            done = True
            # 如果是刚好走到最后一步，给予重罚；如果是在前面就被更换/维修了，则安全
            if action == 0: 
                reward -= self.penalty_break

        # 重新编码新 state
        if not done:
            self.state = self.encode_state()
            
        return self.state, reward, done, {}