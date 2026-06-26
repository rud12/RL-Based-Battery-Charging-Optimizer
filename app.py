from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.envs.battery_env import BatteryEnv
from src.models.actor_critic import ActorCritic
from src.rl.evaluation import evaluate_episode, format_summary
from src.rl.ppo_trainer import PPO
from src.utils.plotting import plot_results


MODEL_PATH = Path("artifacts/battery_ppo_policy.pt")


def make_env(config, randomize_init=False):
    return BatteryEnv(
        init_temp=config["init_temp"],
        init_soc=config["init_soc"],
        threshold_temp=config["threshold_temp"],
        goal_soc=config["goal_soc"],
        ambient_temp=config["ambient_temp"],
        max_steps=config["max_steps"],
        randomize_init=randomize_init,
    )


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_policy(env):
    return ActorCritic(env.obs_dim, env.n_actions)


def save_policy(policy, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), path)


def load_policy(env, path=MODEL_PATH):
    policy = build_policy(env)
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy


def run_evaluation(policy, env, greedy, all_rewards=None, all_success_rates=None):
    traj = evaluate_episode(policy, env, greedy=greedy)
    fig = plot_results(all_rewards or [], all_success_rates or [], traj, env)
    return traj, fig


st.set_page_config(page_title="PPO Battery Charging", layout="wide")
st.title("PPO Battery Charging Optimizer")

with st.sidebar:
    st.header("Battery")
    init_temp = st.number_input("Initial temperature", value=30.0, step=1.0)
    init_soc = st.number_input("Initial SoC", value=20.0, step=1.0, min_value=0.0, max_value=100.0)
    threshold_temp = st.number_input("Threshold temperature", value=55.0, step=1.0)
    goal_soc = st.number_input("Goal SoC", value=80.0, step=1.0, min_value=0.0, max_value=100.0)
    ambient_temp = st.number_input("Ambient temperature", value=25.0, step=1.0)
    max_steps = st.number_input("Max episode steps", value=500, step=10, min_value=1)

    st.header("Training")
    total_steps = st.number_input("Total PPO training steps", value=400_000, step=10_000, min_value=1)
    rollout_steps = st.number_input("Rollout steps", value=1024, step=128, min_value=1)
    learning_rate = st.number_input("Learning rate", value=3e-4, format="%.6f", min_value=0.0)
    seed = st.number_input("Seed", value=0, step=1)
    greedy_eval = st.checkbox("Greedy evaluation", value=True)

config = {
    "init_temp": float(init_temp),
    "init_soc": float(init_soc),
    "threshold_temp": float(threshold_temp),
    "goal_soc": float(goal_soc),
    "ambient_temp": float(ambient_temp),
    "max_steps": int(max_steps),
}

col1, col2 = st.columns(2)
train_clicked = col1.button("Train model", type="primary")
load_clicked = col2.button("Load saved model")

if train_clicked:
    set_seed(int(seed))
    train_env = make_env(config)
    agent = PPO(
        env=train_env,
        lr=float(learning_rate),
        rollout_steps=int(rollout_steps),
    )

    with st.spinner("Training PPO model..."):
        all_rewards, all_success_rates = agent.train(total_steps=int(total_steps), log_interval=10)
        save_policy(agent.policy)

    eval_env = make_env(config)
    traj, fig = run_evaluation(
        agent.policy,
        eval_env,
        greedy=greedy_eval,
        all_rewards=all_rewards,
        all_success_rates=all_success_rates,
    )
    st.session_state["result"] = (traj, fig, eval_env, all_rewards, all_success_rates)
    st.success(f"Model saved to {MODEL_PATH}")

if load_clicked:
    set_seed(int(seed))
    eval_env = make_env(config)
    if not MODEL_PATH.exists():
        st.error(f"No saved model found at {MODEL_PATH}")
    else:
        policy = load_policy(eval_env)
        traj, fig = run_evaluation(policy, eval_env, greedy=greedy_eval)
        st.session_state["result"] = (traj, fig, eval_env, [], [])

if "result" in st.session_state:
    traj, fig, eval_env, _, _ = st.session_state["result"]
    total_reward = sum(traj["rewards"])

    metric_cols = st.columns(5)
    metric_cols[0].metric("Result", traj["result"].upper())
    metric_cols[1].metric("Steps", len(traj["actions"]))
    metric_cols[2].metric("Final SoC", f"{traj['soc'][-1]:.2f}%")
    metric_cols[3].metric("Final Temp", f"{traj['temp'][-1]:.3f} C")
    metric_cols[4].metric("Total Reward", f"{total_reward:.1f}")

    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Evaluation summary"):
        st.code(format_summary(traj, eval_env), language="text")

