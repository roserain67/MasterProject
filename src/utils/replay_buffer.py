import random
import numpy as np
from collections import deque


class EpisodeReplayBuffer:
    """按 episode 存储，支持多任务按 unit 分离采样，支持 n-step return"""

    def __init__(self, train_units, good_capacity=80, recent_capacity=40, context_k=16, n_step=1, gamma=0.99):
        self.train_units = list(train_units)
        self.context_k = context_k
        self.n_step = n_step
        self.gamma = gamma
        self.good_by_unit = {u: deque(maxlen=good_capacity) for u in self.train_units}
        self.recent_by_unit = {u: deque(maxlen=recent_capacity) for u in self.train_units}

    def push_episode(self, transitions, episode_reward, unit_id, good_threshold=120, min_threshold=-3000):
        if len(transitions) < 2 or unit_id not in self.train_units:
            return
        item = (list(transitions), episode_reward)
        if episode_reward >= good_threshold:
            self.good_by_unit[unit_id].append(item)
        if episode_reward >= min_threshold:
            self.recent_by_unit[unit_id].append(item)

    def _sample_nstep(self, ep_transitions, n_samples):
        """从 episode 中采样 n_samples 个 n-step transition"""
        T = len(ep_transitions)
        indices = random.choices(range(T), k=n_samples)
        s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []
        for i in indices:
            s_i, a_i, _, _, _ = ep_transitions[i]
            G = 0.0
            end = min(i + self.n_step, T)
            gamma_k = 1.0
            bootstrap_done = False
            for j in range(i, end):
                _, _, r_j, _, d_j = ep_transitions[j]
                G += gamma_k * r_j
                gamma_k *= self.gamma
                if d_j:
                    bootstrap_done = True
                    s2_final = ep_transitions[j][3]
                    break
            else:
                s2_final = ep_transitions[end - 1][3]
            s_list.append(s_i)
            a_list.append(a_i)
            r_list.append(G)
            s2_list.append(s2_final)
            d_list.append(float(bootstrap_done))
        return (np.array(s_list), np.array(a_list), np.array(r_list),
                np.array(s2_list), np.array(d_list))

    def _context_from_transitions(self, transitions):
        k = min(self.context_k, len(transitions))
        batch = random.sample(transitions, k)
        vecs = [np.concatenate([s, [a, r], s2]) for (s, a, r, s2, d) in batch]
        return np.mean(vecs, axis=0).astype(np.float32)

    def sample(self, batch_size):
        use_good = any(len(self.good_by_unit[u]) > 0 for u in self.train_units) and random.random() < 0.7
        pools = {}
        for u in self.train_units:
            if use_good and len(self.good_by_unit[u]) > 0:
                pools[u] = self.good_by_unit[u]
            elif len(self.recent_by_unit[u]) > 0:
                pools[u] = self.recent_by_unit[u]
            elif len(self.good_by_unit[u]) > 0:
                pools[u] = self.good_by_unit[u]

        available_units = [u for u in self.train_units if u in pools]
        if len(available_units) == 0:
            return None

        if len(available_units) >= 3:
            n_per = batch_size // len(available_units)
            s_list, a_list, r_list, s2_list, d_list, ctx_list = [], [], [], [], [], []
            for u in available_units:
                ep_transitions, _ = random.choice(pools[u])
                if len(ep_transitions) < 2:
                    continue
                n = min(n_per, len(ep_transitions))
                s, a, r, s2, d = self._sample_nstep(ep_transitions, n)
                ctx = self._context_from_transitions(ep_transitions)
                s_list.append(s)
                a_list.append(a)
                r_list.append(r)
                s2_list.append(s2)
                d_list.append(d)
                ctx_list.append((ctx, n))
            if len(ctx_list) >= 2:
                return (np.concatenate(s_list), np.concatenate(a_list), np.concatenate(r_list),
                        np.concatenate(s2_list), np.concatenate(d_list), ctx_list)
            available_units = available_units[:1]

        u = random.choice(available_units)
        ep_transitions, _ = random.choice(pools[u])
        if len(ep_transitions) < 2:
            return None
        n = min(batch_size, len(ep_transitions))
        s, a, r, s2, d = self._sample_nstep(ep_transitions, n)
        ctx = self._context_from_transitions(ep_transitions)
        return s, a, r, s2, d, [(ctx, n)]

    def __len__(self):
        return sum(len(self.good_by_unit[u]) + len(self.recent_by_unit[u]) for u in self.train_units)


class ContextBuffer:
    def __init__(self, capacity=2000, context_k=16):
        self.buffer = deque(maxlen=capacity)
        self.context_k = context_k

    def clear(self):
        self.buffer.clear()

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample_context(self):
        k = min(self.context_k, len(self.buffer))
        if k == 0:
            return None
        batch = random.sample(self.buffer, k)
        vecs = [np.concatenate([s, [a, r], s2]) for (s, a, r, s2, d) in batch]
        return np.mean(vecs, axis=0).astype(np.float32)


class ReplayBuffer:
    """简单 flat replay buffer，用于基线算法 (SAC/TD3/TQC)"""

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
