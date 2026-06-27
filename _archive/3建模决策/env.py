import gym
import numpy as np
from gym import spaces
import torch

class MaintenanceEnv(gym.Env):
    """
    端到端双部件维护环境：
    - state = GRU 输出 (64 维)
    - 输入是某个 unit 的完整 sensor 序列 (seq_len, 172)
    - 维修效果递减：第1次维修回到初始健康，第n次维修恢复效果变差、成本增高
    - 动作7：更换（重置所有，成本高，多次维修后更优）
    """
    metadata = {"render_modes": []}

    def __init__(self, sequence, encoder, state_dim=65, step_size=5, cost_alpha=0.2):
        """
        sequence: numpy array, (seq_len, 172)
        encoder: GRUEncoder 模型
        state_dim: 65 = GRU(64) + 归一化位置(1)，用于区分健康/退化阶段
        step_size: 每次维修后恢复点偏移量（第n次维修回到 cycle=(n-1)*step_size）
        cost_alpha: 维修成本递增系数，第n次成本 = base * (1 + alpha * n)
        """
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

        # reward 参数：0继续运行+15（强化运行），1-3提前维修为正奖励，4-6事后维修为负，7更换为负
        self.task_reward = 15      # 继续运行（提高以强化「健康时运行」）
        self.reward_prev_A = 7     # 提前维修 A（正奖励，鼓励适时维护）
        self.reward_prev_B = 7     # 提前维修 B
        self.reward_prev_AB = 5    # 提前维修 A+B（略低，因范围更大）
        self.cost_post_A = 13
        self.cost_post_B = 13
        self.cost_post_AB = 15
        self.cost_replace = 20     # 更换成本

        self.step_size = step_size
        self.cost_alpha = cost_alpha

        # 内部指针与维修计数
        self.pointer = 0
        self.repair_count_A = 0
        self.repair_count_B = 0
        self.repair_in_row = 0   # 连续维修次数，用于早停
        self.max_repair_in_row = 15  # 连续维修超过此次数则 done 并惩罚
        self.state = None

    def _restore_point(self, count):
        """第 count 次维修后的恢复点：第1次=0，第n次=(n-1)*step_size"""
        return min((count - 1) * self.step_size, self.seq_len - 1)

    def _scaled_cost(self, base_cost, count):
        """成本递增：第n次维修成本 = base * (1 + alpha * n)"""
        return base_cost * (1 + self.cost_alpha * count)

    # ---------------- GRU 编码 state ---------------- #
    def encode_state(self):
        """
        当前 pointer 对应的序列片段输入 GRU → state embedding
        追加归一化位置 pointer/seq_len，使策略能区分健康/退化阶段
        """
        if self.pointer == 0:
            seq = self.sequence[:1]  # (1,172)
        else:
            seq = self.sequence[: self.pointer + 1]  # (t,172)

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).cuda()  # (1, t, 172)
        with torch.no_grad():
            emb = self.encoder(seq_tensor)  # (1, 64)
        emb_np = emb.squeeze(0).cpu().numpy()  # (64,)
        # 追加归一化位置：pointer 小=健康，pointer 大=退化
        pos = np.array([self.pointer / max(1, self.seq_len)], dtype=np.float32)
        return np.concatenate([emb_np, pos])  # (65,)

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

        # 0：继续运行（pointer + 1）
        if action == 0:
            self.repair_in_row = 0
            reward += self.task_reward
            self.pointer += 1

        # 1：提前维修 A（正奖励，略低于继续运行）
        elif action == 1:
            self.repair_in_row += 1
            self.repair_count_A += 1
            r = self.reward_prev_A - 0.3 * (self.repair_count_A - 1)  # 多次维修略递减
            reward += max(1.0, r)
            self.pointer = self._restore_point(self.repair_count_A)

        # 2：提前维修 B
        elif action == 2:
            self.repair_in_row += 1
            self.repair_count_B += 1
            r = self.reward_prev_B - 0.3 * (self.repair_count_B - 1)
            reward += max(1.0, r)
            self.pointer = self._restore_point(self.repair_count_B)

        # 3：提前维修 A+B
        elif action == 3:
            self.repair_in_row += 1
            self.repair_count_A += 1
            self.repair_count_B += 1
            r = self.reward_prev_AB - 0.3 * (max(self.repair_count_A, self.repair_count_B) - 1)
            reward += max(1.0, r)
            rp_A = self._restore_point(self.repair_count_A)
            rp_B = self._restore_point(self.repair_count_B)
            self.pointer = max(rp_A, rp_B)

        # 4：事后维修 A
        elif action == 4:
            self.repair_in_row += 1
            self.repair_count_A += 1
            cost = self._scaled_cost(self.cost_post_A, self.repair_count_A)
            reward -= (cost + self.task_reward)
            self.pointer = self._restore_point(self.repair_count_A)

        # 5：事后维修 B
        elif action == 5:
            self.repair_in_row += 1
            self.repair_count_B += 1
            cost = self._scaled_cost(self.cost_post_B, self.repair_count_B)
            reward -= (cost + self.task_reward)
            self.pointer = self._restore_point(self.repair_count_B)

        # 6：事后维修 A+B
        elif action == 6:
            self.repair_in_row += 1
            self.repair_count_A += 1
            self.repair_count_B += 1
            cost = self._scaled_cost(self.cost_post_AB, max(self.repair_count_A, self.repair_count_B))
            reward -= (cost + self.task_reward)
            rp_A = self._restore_point(self.repair_count_A)
            rp_B = self._restore_point(self.repair_count_B)
            self.pointer = max(rp_A, rp_B)

        # 7：更换（重置到初始健康，维修计数归零）
        elif action == 7:
            self.repair_in_row = 0
            reward -= self.cost_replace
            self.repair_count_A = 0
            self.repair_count_B = 0
            self.pointer = 0

        # ---------------- 连续维修早停 ----------------
        if self.repair_in_row >= self.max_repair_in_row:
            done = True
            reward -= 50  # 过度维修惩罚

        # ---------------- 检查是否越界 ---------------- #
        if self.pointer >= self.seq_len:
            done = True
            reward -= 20  # 到达序列末尾（退化无法继续）

        # ---------------- 重新编码新 state ---------------- #
        self.state = self.encode_state()
        return self.state, reward, done, {}

