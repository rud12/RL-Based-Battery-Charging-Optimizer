import torch
from torch.distributions import Categorical


def evaluate_episode(policy, env, greedy=True):
    obs = env.reset()
    done = False
    traj = {"soc": [env.soc], "temp": [env.temp], "actions": [], "rewards": []}

    while not done:
        x = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, _ = policy(x)
        if greedy:
            action = logits.argmax(dim=-1).item()
        else:
            action = Categorical(logits=logits).sample().item()
        obs, reward, done, info = env.step(action)
        traj["soc"].append(env.soc)
        traj["temp"].append(env.temp)
        traj["actions"].append(env.C_RATES[action])
        traj["rewards"].append(reward)

    traj["result"] = info["result"]
    return traj

