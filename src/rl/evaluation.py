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


def format_summary(traj, env):
    lines = [
        f"Result      : {traj['result'].upper()}",
        f"Steps taken : {len(traj['actions'])}  (1 step = 1 min)",
        f"Final SoC   : {traj['soc'][-1]:.2f}%   (goal={env.goal_soc}%)",
        f"Final Temp  : {traj['temp'][-1]:.3f}C  (threshold={env.threshold_temp}C)",
        f"Total reward: {sum(traj['rewards']):.1f}",
        "",
        f"{'Min':>4}  {'C-rate':>7}  {'SoC (%)':>8}  {'Temp (C)':>10}  Note",
        f"{'-' * 4}  {'-' * 7}  {'-' * 8}  {'-' * 10}  ----",
    ]
    prev_c = None
    for i, (c, soc, tmp) in enumerate(
        zip(traj["actions"], traj["soc"][1:], traj["temp"][1:])
    ):
        note = ""
        if c != prev_c:
            note = f"<- C-rate change to {c}C"
        if tmp >= env.threshold_temp:
            note = "<- OVERHEAT"
        if soc >= env.goal_soc:
            note = "<- GOAL REACHED"
        lines.append(f"{i + 1:>4}  {c:>7.1f}  {soc:>8.2f}  {tmp:>10.4f}  {note}")
        prev_c = c
    return "\n".join(lines)


def print_summary(traj, env):
    print(f"\n{'=' * 62}")
    print(" Greedy Evaluation - Step-by-Step Summary")
    print(f"{'=' * 62}")
    print(format_summary(traj, env))
    print(f"{'=' * 62}\n")

