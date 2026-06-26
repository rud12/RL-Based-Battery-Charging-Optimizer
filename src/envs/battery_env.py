import numpy as np


class BatteryEnv:
    """
    Hardcoded battery model with realistic thermal dynamics.

    State : [soc/100, temp/T_threshold, headroom/(T_threshold - T_ambient)]
    Action: index into C_RATES
    """

    C_RATES = [0.5, 1.0, 2.0, 3.0]
    DELTA_SOC = [0.40, 0.80, 1.60, 3.60]
    HEAT_GEN = [0.10, 0.25, 0.60, 1.40]
    COOLING_COEF = 0.002

    def __init__(
        self,
        init_temp=30.0,
        init_soc=20.0,
        threshold_temp=55.0,
        goal_soc=80.0,
        ambient_temp=25.0,
        max_steps=500,
        randomize_init=False,
    ):
        self.init_temp = init_temp
        self.init_soc = init_soc
        self.threshold_temp = threshold_temp
        self.goal_soc = goal_soc
        self.ambient_temp = ambient_temp
        self.max_steps = max_steps
        self.randomize_init = randomize_init

        self.n_actions = len(self.C_RATES)
        self.obs_dim = 3

        self.reset()

    def reset(self):
        if self.randomize_init:
            self.soc = float(np.random.uniform(self.init_soc - 5, self.init_soc + 5))
            self.temp = float(np.random.uniform(self.init_temp - 2, self.init_temp + 2))
        else:
            self.soc = float(self.init_soc)
            self.temp = float(self.init_temp)

        self.soc = np.clip(self.soc, 0.0, 100.0)
        self.steps = 0
        return self._obs()

    def _obs(self):
        soc_norm = self.soc / 100.0
        temp_norm = self.temp / self.threshold_temp
        headroom = (self.threshold_temp - self.temp) / max(
            self.threshold_temp - self.ambient_temp, 1e-6
        )
        headroom = np.clip(headroom, -0.5, 1.0)
        return np.array([soc_norm, temp_norm, headroom], dtype=np.float32)

    def step(self, action: int):
        c_rate = self.C_RATES[action]
        delta_soc = self.DELTA_SOC[action]
        heat_gen = self.HEAT_GEN[action]

        passive_cooling = self.COOLING_COEF * (self.temp - self.ambient_temp)
        delta_T = heat_gen - passive_cooling
        self.temp += delta_T
        self.soc = min(self.soc + delta_soc, 100.0)
        self.steps += 1

        headroom = self.threshold_temp - self.temp
        soc_frac = self.soc / 100.0

        reward = 10.0 * delta_soc
        reward -= 0.8

        if headroom > 15.0:
            reward += 2.0 * c_rate
        elif headroom > 8.0:
            reward += 0.8 * c_rate
        else:
            reward -= 4.0 * c_rate

        reward += 2.0 * (1.0 - soc_frac) * delta_soc

        if headroom < 10.0:
            frac = (10.0 - headroom) / 10.0
            reward -= 25.0 * (frac ** 2)

        done = False

        if self.temp >= self.threshold_temp:
            reward -= 250.0
            done = True
            info = {"result": "overheat"}
        elif self.soc >= self.goal_soc:
            reward += 300.0 + 5.0 * headroom
            done = True
            info = {"result": "success"}
        elif self.steps >= self.max_steps:
            reward -= 20.0
            done = True
            info = {"result": "timeout"}
        else:
            info = {"result": "running"}

        return self._obs(), reward, done, info

