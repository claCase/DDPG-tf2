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
        self.td_errors = np.zeros(shape=(self.max_size, 1))

    def store_transition(self, s0, a, r, s1, d, td):
        self.mem_cntr = self.mem_cntr % self.max_size
        if td > np.quantile(self.td_errors, 0.1, axis=0):
            self.s0[self.mem_cntr] = s0
            self.s1[self.mem_cntr] = s1
            self.a[self.mem_cntr] = a
            self.r[self.mem_cntr] = r
            self.d[self.mem_cntr] = d
            self.td_errors[self.mem_cntr] = td
            self.mem_cntr += 1

    def sample(self, batch_size=None):
        if batch_size is None:
            b = self.batch_size
        else:
            b = batch_size
        if self.mem_cntr > 1:
            b = np.minimum(b, self.mem_cntr)
            d = self.d - np.max(self.d)
            p = np.exp(d*self.alpha) / np.sum(np.exp(d * self.alpha))
            idx = np.random.choice(p.shape[0]+1, p=p, size=b, replace=False)
            w = np.pow(b*p[idx], self.beta)
            w = w/np.max(w)
            return self.s0[idx], self.a[idx], self.r[idx], self.s1[idx], self.d[idx], w
        else:
            w = 1.0
            return self.s0[:1], self.a[:1], self.r[:1], self.s1[:1], self.d[:1], w