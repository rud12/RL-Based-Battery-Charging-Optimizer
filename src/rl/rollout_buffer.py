import numpy as np
import torch


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_gae(self, last_value: float, gamma: float, lam: float):
        T = len(self.rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_value if t == T - 1 else self.values[t + 1]
            next_done = float(self.dones[t])
            delta = (
                self.rewards[t]
                + gamma * next_val * (1.0 - next_done)
                - self.values[t]
            )
            last_gae = delta + gamma * lam * (1.0 - next_done) * last_gae
            adv[t] = last_gae
        returns = adv + np.array(self.values, dtype=np.float32)
        return torch.FloatTensor(adv), torch.FloatTensor(returns)

    def tensors(self):
        return (
            torch.FloatTensor(np.array(self.obs)),
            torch.LongTensor(self.actions),
            torch.FloatTensor(self.log_probs),
        )

