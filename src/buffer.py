import numpy as np
import os


class Buffer:
    def __init__(self, state_shape, action_shape, reward_shape, max_size, batch_size):
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.state_shape = int(state_shape)
        self.action_shape = int(action_shape)
        self.reward_shape = int(reward_shape)
        self.s0 = np.zeros(shape=(self.max_size, state_shape), dtype=np.float32)
        self.s1 = np.zeros(shape=(self.max_size, state_shape), dtype=np.float32)
        self.a = np.zeros(shape=(self.max_size, action_shape), dtype=np.float32)
        self.r = np.zeros(shape=(self.max_size, reward_shape), dtype=np.float32)
        self.d = np.zeros(shape=(self.max_size, 1), dtype=np.float32)
        self.mem_cntr = 0

    def store_transition(self, s0, a, r, s1, d):
        self.mem_cntr = self.mem_cntr % self.max_size
        self.s0[self.mem_cntr] = s0
        self.s1[self.mem_cntr] = s1
        self.a[self.mem_cntr] = a
        self.r[self.mem_cntr] = r
        self.d[self.mem_cntr] = d
        self.mem_cntr += 1

    def sample(self, batch_size=None):
        if batch_size is None:
            b = self.batch_size
        else:
            b = batch_size
        b = np.minimum(b, self.mem_cntr)
        idx = np.random.choice(self.mem_cntr, b, False)
        return self.s0[idx], self.a[idx], self.r[idx], self.s1[idx], self.d[idx]

    def save(self, dir):
        if not os.path.exists(os.path.join(dir, "buffer")):
            os.mkdir(os.path.join(dir, "buffer"))
        np.save(os.path.join(dir, "buffer", "s0"), self.s0)
        np.save(os.path.join(dir, "buffer", "s1"), self.s1)
        np.save(os.path.join(dir, "buffer", "a"), self.a)
        np.save(os.path.join(dir, "buffer", "r"), self.r)
        np.save(os.path.join(dir, "buffer", "d"), self.d)

    def load(self, dir):
        self.s0 = np.load(os.path.join(dir, "buffer", "s0.npy"))
        self.s1 = np.load(os.path.join(dir, "buffer", "s1.npy"))
        self.a = np.load(os.path.join(dir, "buffer", "a.npy"))
        self.r = np.load(os.path.join(dir, "buffer", "r.npy"))
        self.d = np.load(os.path.join(dir, "buffer", "d.npy"))


class PriorityBuffer(Buffer):
    def __init__(self, alpha, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros(shape=(self.max_size, 1))
        self.priorities[0] = 1.0
        self.mem_cntr = 0

    def store_transition(self, s0, a, r, s1, d):
        self.mem_cntr = self.mem_cntr % self.max_size
        self.s0[self.mem_cntr] = s0
        self.s1[self.mem_cntr] = s1
        self.a[self.mem_cntr] = a
        self.r[self.mem_cntr] = r
        self.d[self.mem_cntr] = d
        if self.mem_cntr != 0:
            self.priorities[self.mem_cntr] = np.max(self.priorities)
        self.mem_cntr += 1

    def update_priority(self, priorities, indices):
        self.priorities[indices] = np.abs(priorities)

    def sample(self, batch_size=None):
        if batch_size is None:
            b = self.batch_size
        else:
            b = batch_size
        b = np.minimum(b, self.mem_cntr)
        d = (self.priorities[:self.mem_cntr]) ** self.alpha
        p = d / np.sum(d)
        # print(p)
        idx = np.random.choice(a=p.shape[0], p=p[:, 0], size=b, replace=True)
        w = np.power(b * p[idx], -self.beta)
        w = w / np.max(w)
        w = w.astype(np.float32)
        return self.s0[idx], self.a[idx], self.r[idx], self.s1[idx], self.d[idx], w, idx
