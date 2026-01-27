import numpy as np

class ReplayMemory:
    """
    基础 Buffer 类 (单池)，负责底层的存储和切片操作。
    """
    def __init__(self, buffer_capacity=100000, batch_size=24):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        # 初始化 Buffer (使用 empty 预分配内存)
        self.state_buffer = np.empty(self.buffer_capacity, dtype=object)
        self.action_buffer = np.empty(self.buffer_capacity, dtype=object)
        self.next_state_buffer = np.empty(self.buffer_capacity, dtype=object)
        self.reward_buffer = np.zeros(self.buffer_capacity, dtype=np.float32)
        self.done_buffer = np.zeros(self.buffer_capacity, dtype=np.bool_)

    def record(self, state, action, rew, next_state, done):
        """写入数据 (内部方法，不直接处理 is_expert)"""
        # 判断是单条还是批量
        if np.isscalar(rew):
            num_entries = 1
        else:
            num_entries = len(rew)

        start_idx = self.buffer_counter % self.buffer_capacity
        end_idx = start_idx + num_entries

        if end_idx <= self.buffer_capacity:
            self._insert_batch(start_idx, end_idx, state, action, rew, next_state, done)
        else:
            # 处理回卷
            mid = self.buffer_capacity - start_idx
            self._insert_batch(start_idx, self.buffer_capacity,
                               state[:mid], action[:mid], rew[:mid], next_state[:mid], done[:mid])
            rem = num_entries - mid
            self._insert_batch(0, rem,
                               state[mid:], action[mid:], rew[mid:], next_state[mid:], done[mid:])

        self.buffer_counter += num_entries

    def _insert_batch(self, start, end, s, a, r, ns, d):
        # 针对单条数据的特殊处理，防止 slice 维度错误
        if end - start == 1 and not isinstance(r, (list, np.ndarray)):
             self.state_buffer[start] = s
             self.action_buffer[start] = a
             self.reward_buffer[start] = r
             self.next_state_buffer[start] = ns
             self.done_buffer[start] = d
        else:
             self.state_buffer[start:end] = s
             self.action_buffer[start:end] = a
             self.reward_buffer[start:end] = r
             self.next_state_buffer[start:end] = ns
             self.done_buffer[start:end] = d

    def sample(self, batch_size=None):
        """支持指定 batch_size"""
        if batch_size is None:
            batch_size = self.batch_size

        curr_size = min(self.buffer_counter, self.buffer_capacity)
        if curr_size == 0:
            return None

        idxes = np.random.randint(0, curr_size, batch_size)

        batch = dict(state=self.state_buffer[idxes],
                    action=self.action_buffer[idxes],
                    reward=self.reward_buffer[idxes],
                    next_state=self.next_state_buffer[idxes],
                    done = self.done_buffer[idxes])
        return batch

    def __len__(self):
        return min(self.buffer_counter, self.buffer_capacity)


class DualReplayMemory:
    """
    [核心修改] 双缓冲管理类
    自动管理 Expert Buffer (BFGS) 和 Agent Buffer (RL)，并按比例混合采样。
    """
    def __init__(self, buffer_capacity=100000, batch_size=24, expert_ratio=0.2):
        self.batch_size = batch_size
        self.expert_ratio = expert_ratio

        # 两个独立的池子
        # 专家池容量可以设小一点，或者设为一样大，这里设为一样大以防溢出
        self.expert_mem = ReplayMemory(buffer_capacity=buffer_capacity, batch_size=batch_size)
        self.agent_mem = ReplayMemory(buffer_capacity=buffer_capacity, batch_size=batch_size)

    def record(self, state, action, rew, next_state, done, is_expert=False):
        """
        根据 is_expert 标志分流数据
        """
        if is_expert:
            self.expert_mem.record(state, action, rew, next_state, done)
        else:
            self.agent_mem.record(state, action, rew, next_state, done)

    def sample(self):
        """
        混合采样：20% 专家 + 80% 自身
        """
        n_expert = int(self.batch_size * self.expert_ratio)
        n_agent = self.batch_size - n_expert

        # 分别采样
        batch_expert = self.expert_mem.sample(batch_size=n_expert)
        batch_agent = self.agent_mem.sample(batch_size=n_agent)

        # --- 容错处理 (处理前期某个池子可能为空的情况) ---
        if batch_agent is None and batch_expert is None:
            # 两个池子都空，无法采样
            return None

        elif batch_agent is None:
            # 只有专家数据 (Warmup 刚结束，Agent还没怎么跑)
            return self.expert_mem.sample(self.batch_size)

        elif batch_expert is None:
            # 只有普通数据 (几乎不可能发生，因为有 Warmup)
            return self.agent_mem.sample(self.batch_size)

        # --- 拼接数据 ---
        combined_batch = {}
        for key in batch_expert.keys():
            # Concatenate arrays along axis 0
            combined_batch[key] = np.concatenate([batch_expert[key], batch_agent[key]], axis=0)

        return combined_batch

    def __len__(self):
        return len(self.expert_mem) + len(self.agent_mem)