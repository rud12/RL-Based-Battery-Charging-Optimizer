import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def plot_results(all_rewards, all_success_rates, traj, env, save_path=None):
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    c_colors = {0.5: "#2ecc71", 1.0: "#3498db", 2.0: "#e67e22", 3.0: "#e74c3c"}
    steps_t = list(range(len(traj["soc"])))

    ax1 = fig.add_subplot(gs[0, 0])
    if all_rewards:
        w = min(80, len(all_rewards))
        sm = np.convolve(all_rewards, np.ones(w) / w, mode="valid")
        ax1.plot(all_rewards, alpha=0.18, color="steelblue", linewidth=0.5)
        ax1.plot(
            range(w - 1, len(all_rewards)),
            sm,
            color="steelblue",
            linewidth=2.0,
            label=f"{w}-ep avg",
        )
        ax1.legend(fontsize=8)
    ax1.set_title("Training Episode Reward", fontweight="bold")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(gs[0, 1])
    if all_success_rates:
        ax2.plot([r * 100 for r in all_success_rates], color="#27ae60", linewidth=1.5)
    ax2.set_ylim(-5, 108)
    ax2.axhline(100, color="#27ae60", linestyle="--", alpha=0.4)
    ax2.set_title("Success Rate (rolling 30 eps)", fontweight="bold")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success %")
    ax2.grid(alpha=0.25)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(steps_t, traj["soc"], color="#2980b9", linewidth=2.2)
    ax3.axhline(
        env.goal_soc,
        color="#2980b9",
        linestyle="--",
        alpha=0.55,
        label=f"Goal {env.goal_soc}%",
    )
    ax3.axhline(
        env.init_soc,
        color="gray",
        linestyle=":",
        alpha=0.45,
        label=f"Init {env.init_soc}%",
    )
    ax3.set_title(f"SoC Trajectory  [{traj['result'].upper()}]", fontweight="bold")
    ax3.set_xlabel("Time (min)")
    ax3.set_ylabel("SoC (%)")
    ax3.set_ylim(0, 105)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(steps_t, traj["temp"], color="#e74c3c", linewidth=2.2)
    ax4.axhline(
        env.threshold_temp,
        color="#c0392b",
        linestyle="--",
        alpha=0.8,
        label=f"Threshold {env.threshold_temp}C",
    )
    ax4.axhline(
        env.init_temp,
        color="#e67e22",
        linestyle=":",
        alpha=0.55,
        label=f"Init {env.init_temp}C",
    )
    ax4.fill_between(
        steps_t, env.threshold_temp - 8, env.threshold_temp, alpha=0.08, color="red", label="Penalty zone"
    )
    ax4.set_title("Temperature Trajectory", fontweight="bold")
    ax4.set_xlabel("Time (min)")
    ax4.set_ylabel("Temperature (C)")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.25)

    ax5 = fig.add_subplot(gs[1, 1])
    for i, c in enumerate(traj["actions"]):
        ax5.bar(i, c, color=c_colors.get(c, "gray"), width=1.0, alpha=0.75)
    ax5.set_title("C-rate Selected per Minute", fontweight="bold")
    ax5.set_xlabel("Time (min)")
    ax5.set_ylabel("C-rate")
    ax5.set_yticks(env.C_RATES)
    ax5.grid(alpha=0.25, axis="y")
    handles = [Patch(color=v, label=f"{k}C") for k, v in c_colors.items()]
    ax5.legend(handles=handles, fontsize=8, loc="upper right")

    ax6 = fig.add_subplot(gs[1, 2])
    sc = ax6.scatter(
        traj["soc"], traj["temp"], c=range(len(traj["soc"])), cmap="plasma", s=18, zorder=3
    )
    for i in range(len(traj["soc"]) - 1):
        ax6.plot(
            traj["soc"][i : i + 2],
            traj["temp"][i : i + 2],
            color="gray",
            alpha=0.3,
            linewidth=0.8,
        )
    ax6.axhline(
        env.threshold_temp,
        color="#c0392b",
        linestyle="--",
        alpha=0.7,
        label=f"T threshold {env.threshold_temp}C",
    )
    ax6.axvline(
        env.goal_soc,
        color="#2980b9",
        linestyle="--",
        alpha=0.7,
        label=f"SoC goal {env.goal_soc}%",
    )
    plt.colorbar(sc, ax=ax6, label="Time (min)")
    ax6.set_title("Phase Portrait: SoC vs Temperature", fontweight="bold")
    ax6.set_xlabel("SoC (%)")
    ax6.set_ylabel("Temperature (C)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.25)

    fig.suptitle(
        "PPO Battery Charging Optimizer - Physics-Correct Thermal Model",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

