import numpy as np
import torch
import torch.nn as nn

# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Expects tuples of (state, next_state, action, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.const_w = None

    def add(self, state, new_state, action, reward, done_bool):
        data = (np.array(state, copy=False), np.array(new_state, copy=False),
                np.array(action, copy=False), np.array(reward, copy=False),
                    np.array(done_bool, copy=False))
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size, beta):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(X)
            y.append(Y)
            u.append(U)
            r.append(R)
            d.append(D)


        #print "X.shape = ", np.array(U).shape, "x.shape = ", np.array(u).shape
        return (np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1),
                np.array(d).reshape(-1, 1), None, np.ones((batch_size,), dtype=np.float32))

    def update_priorities(self, x, y):
        pass

class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        # Buffers to reuse memory
        self.states = None
        self.next_states = None

    def add(self, state, next_state, action, reward, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, next_state, action, reward, done))
        else:
            self.buffer[self.pos] = (state, next_state, action, reward, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Get the weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        samples = [self.buffer[idx] for idx in indices]
        batch = list(zip(*samples))

        if self.states is None:
            self.states = np.array(np.concatenate(batch[0]), copy=False)
            self.next_states = np.array(np.concatenate(batch[1]), copy=False)
        else:
            np.concatenate(batch[0], out=self.states)
            np.concatenate(batch[1], out=self.next_states)

        actions = np.array(batch[2], copy=False)
        rewards = np.array(batch[3], copy=False).reshape(-1, 1)
        dones = np.array(batch[4], copy=False).reshape(-1, 1)

        return (self.states, self.next_states, actions, rewards, dones,
            indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in list(zip(batch_indices, batch_priorities)):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

