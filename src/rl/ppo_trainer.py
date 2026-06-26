from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.actor_critic import ActorCritic
from src.rl.rollout_buffer import RolloutBuffer


class PPO:
    def __init__(
        self,
        env,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        epochs=8,
        batch_size=128,
        vf_coef=0.5,
        ent_coef=0.02,
        max_grad_norm=0.5,
        rollout_steps=1024,
    ):
        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps = rollout_steps

        self.policy = ActorCritic(env.obs_dim, env.n_actions)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def collect_rollout(self, obs):
        self.buffer.clear()
        ep_rewards, ep_results = [], []
        ep_r = 0.0

        for _ in range(self.rollout_steps):
            action, log_prob, value = self.policy.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.buffer.add(obs, action, log_prob, reward, done, value)
            ep_r += reward
            obs = next_obs

            if done:
                ep_rewards.append(ep_r)
                ep_results.append(info["result"])
                ep_r = 0.0
                obs = self.env.reset()

        with torch.no_grad():
            _, last_value = self.policy(torch.FloatTensor(obs).unsqueeze(0))
            last_value = last_value.item()

        adv, returns = self.buffer.compute_returns_gae(last_value, self.gamma, self.lam)
        return obs, adv, returns, ep_rewards, ep_results

    def update(self, adv, returns):
        obs_t, act_t, old_lp_t = self.buffer.tensors()
        adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = len(obs_t)
        losses = []
        for _ in range(self.epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start : start + self.batch_size]
                lp, vals, ent = self.policy.evaluate(obs_t[mb], act_t[mb])

                ratio = torch.exp(lp - old_lp_t[mb])
                s1 = ratio * adv_norm[mb]
                s2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * adv_norm[mb]
                )
                a_loss = -torch.min(s1, s2).mean()
                c_loss = nn.functional.mse_loss(vals, returns[mb])
                e_loss = -ent.mean()
                loss = a_loss + self.vf_coef * c_loss + self.ent_coef * e_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.opt.step()
                losses.append(loss.item())

        return np.mean(losses)

    def train(self, total_steps: int = 300_000, log_interval: int = 10):
        obs = self.env.reset()
        update_idx = 0
        steps_done = 0
        all_rewards, all_success_rates = [], []
        reward_win = deque(maxlen=30)
        success_win = deque(maxlen=30)

        print(f"\n{'=' * 60}")
        print(" Battery Charging PPO - Physics-Correct Thermal Model")
        print(f"{'=' * 60}")
        print(f" C-rates     : {self.env.C_RATES}")
        print(f" DeltaSoC/step   : {self.env.DELTA_SOC}  (%/min)")
        print(f" DeltaT_heat/step: {self.env.HEAT_GEN}  (C/min)")
        print(f" Cooling coef: {self.env.COOLING_COEF}  (C/min/C above ambient)")
        print(f" Init  SoC={self.env.init_soc}%  Temp={self.env.init_temp}C")
        print(f" Goal  SoC={self.env.goal_soc}%  Threshold={self.env.threshold_temp}C")
        print(
            " Thermal headroom at start: "
            f"{self.env.threshold_temp - self.env.init_temp:.1f}C"
        )
        print(f"{'=' * 60}\n")

        while steps_done < total_steps:
            obs, adv, returns, ep_rewards, ep_results = self.collect_rollout(obs)
            loss = self.update(adv, returns)
            steps_done += self.rollout_steps
            update_idx += 1

            for r, res in zip(ep_rewards, ep_results):
                reward_win.append(r)
                success_win.append(1.0 if res == "success" else 0.0)
                all_rewards.append(r)
                all_success_rates.append(np.mean(success_win))

            if update_idx % log_interval == 0:
                mean_r = np.mean(reward_win) if reward_win else 0.0
                succ_pct = 100 * np.mean(success_win) if success_win else 0.0
                print(
                    f" Update {update_idx:4d} | Steps {steps_done:8d} "
                    f"| MeanRew {mean_r:8.1f} | Success {succ_pct:5.1f}% | Loss {loss:.4f}"
                )

        print("\n Training complete.\n")
        return all_rewards, all_success_rates

