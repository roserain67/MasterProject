import gym
import numpy as np
from gym import spaces
import torch

class MaintenanceEnv(gym.Env):
    """
    端到端双部件维护环境：
    - state = GRU 输出 (64 维)
    - 输入是某个 unit 的完整 sensor 序列 (seq_len, 172)
    - 无 RUL，无健康度衰退，全依赖真实传感器序列
    """

    metadata = {"render_modes": []}

    def __init__(self, sequence, encoder, state_dim=64):
        """
        sequence: numpy array, (seq_len, 172)
        encoder: GRUEncoder 模型
        """
        super().__init__()

        self.sequence = sequence.astype(np.float32)
        self.seq_len = sequence.shape[0]
        self.encoder = encoder.eval().cuda()  # 你已有的 GRU 模型

        self.state_dim = state_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # 7 个动作
        self.action_space = spaces.Discrete(7)

        # reward 参数
        self.task_reward = 10
        self.cost_prev_A = 3
        self.cost_prev_B = 3
        self.cost_prev_AB = 5
        self.cost_post_A = 13
        self.cost_post_B = 13
        self.cost_post_AB = 15

        # 内部指针
        self.pointer = 0  # 当前 cycle index
        self.state = None

    # ---------------- GRU 编码 state ---------------- #
    def encode_state(self):
        """
        当前 pointer 对应的序列片段输入 GRU → state embedding
        采用前 pointer+1 个 cycle 的前缀序列进行编码
        """
        if self.pointer == 0:
            seq = self.sequence[:1]  # (1,172)
        else:
            seq = self.sequence[: self.pointer + 1]  # (t,172)

        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).cuda()  # (1, t, 172)
        with torch.no_grad():
            emb = self.encoder(seq_tensor)  # (1, 64)
        return emb.squeeze(0).cpu().numpy()  # (64,)

    # ---------------- reset ---------------- #
    def reset(self):
        self.pointer = 0
        self.state = self.encode_state()
        return self.state

    # ---------------- step ---------------- #
    def step(self, action):
        reward = 0
        done = False

        # 0：继续运行（pointer + 1）
        if action == 0:
            reward += self.task_reward
            self.pointer += 1

        # 1：提前维修 A（回到最初点）
        elif action == 1:
            reward -= self.cost_prev_A
            self.pointer = 0

        # 2：提前维修 B
        elif action == 2:
            reward -= self.cost_prev_B
            self.pointer = 0

        # 3：提前维修 A+B
        elif action == 3:
            reward -= self.cost_prev_AB
            self.pointer = 0

        # 4：事后维修 A，会耽误一次飞行
        elif action == 4:
            reward -= (self.cost_post_A + self.task_reward)
            self.pointer = 0

        # 5：事后维修 B
        elif action == 5:
            reward -= (self.cost_post_B + self.task_reward)
            self.pointer = 0

        # 6：事后维修 A+B
        elif action == 6:
            reward -= (self.cost_post_AB + self.task_reward)
            self.pointer = 0

        # ---------------- 检查是否越界 ---------------- #
        if self.pointer >= self.seq_len:
            done = True
            reward -= 20  # 到达序列末尾（退化无法继续）

        # ---------------- 重新编码新 state ---------------- #
        self.state = self.encode_state()
        return self.state, reward, done, {}

