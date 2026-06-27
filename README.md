# 🔋 RL-Based Battery Charging Optimizer

A reinforcement learning agent trained with **Proximal Policy Optimization (PPO)** to optimally charge a battery — maximizing the State of Charge (SoC) while preventing thermal overheating. Built with PyTorch and served through an interactive Streamlit dashboard.

---

## 📖 Table of Contents

- [Project Description](#project-description)
- [How It Works](#how-it-works)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Steps to Run](#steps-to-run)
- [Configuration Parameters](#configuration-parameters)
- [PPO Hyperparameters](#ppo-hyperparameters)
- [Environment Details](#environment-details)
- [Reward Function](#reward-function)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)

---

## Project Description

This project implements a **Reinforcement Learning (RL)** controller for battery charging using a physics-inspired thermal model. The agent learns to select the optimal **C-rate** (charging current rate) at each timestep to:

- ✅ Reach the target **State of Charge (SoC)** as quickly as possible
- 🌡️ Keep the battery **temperature below a safety threshold**
- ⚖️ Balance charging speed vs. thermal risk

The charging policy is trained entirely from scratch using **PPO** — a state-of-the-art on-policy RL algorithm — and evaluated in real time through an interactive **Streamlit web dashboard**.

---

## How It Works

```
┌────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                  │
│  ┌─────────────┐               ┌────────────────────┐  │
│  │  Sidebar    │               │  Results Panel     │  │
│  │  - Battery  │               │  - Metrics         │  │
│  │    config   │               │  - 6-panel plots   │  │
│  │  - Training │               │    (SoC, Temp,     │  │
│  │    config   │               │     C-rate, Phase) │  │
│  └─────────────┘               └────────────────────┘  │
└────────────────────────────────────────────────────────┘
           │ Train / Load
           ▼
┌──────────────────────┐      ┌──────────────────────────┐
│     PPO Trainer      │─────▶│     BatteryEnv           │
│  - Rollout collect   │      │  - Thermal dynamics      │
│  - GAE computation   │      │  - SoC update            │
│  - Policy update     │      │  - Reward shaping        │
└──────────────────────┘      └──────────────────────────┘
           │
           ▼
┌──────────────────────┐
│  ActorCritic (PyTorch│
│  3-layer shared MLP  │
│  Actor: policy head  │
│  Critic: value head  │
└──────────────────────┘
```

At each timestep the agent observes a **3-dimensional state vector**:

| Feature | Description |
|---|---|
| `soc / 100` | Normalized State of Charge |
| `temp / threshold_temp` | Normalized battery temperature |
| `headroom / (T_threshold - T_ambient)` | Normalized thermal headroom |

And chooses one of **4 discrete actions** (C-rates):

| Action | C-rate | ΔSoC/step (%) | Heat generated/step (°C) |
|---|---|---|---|
| 0 | 0.5C | 0.40 | 0.10 |
| 1 | 1.0C | 0.80 | 0.25 |
| 2 | 2.0C | 1.60 | 0.60 |
| 3 | 3.0C | 3.60 | 1.40 |

---

## Features

- 🧠 **Custom PPO implementation** from scratch using PyTorch
- 🌡️ **Physics-inspired thermal model** with passive cooling and heat generation per C-rate
- 📊 **Rich visualization dashboard** — 6-panel Matplotlib plots including SoC trajectory, temperature trajectory, C-rate bar chart, and a SoC–Temperature phase portrait
- 💾 **Model persistence** — trained policies are saved to `artifacts/` and can be reloaded later
- 📈 **Live training progress** — real-time progress bar and success rate metrics during training

---

## Project Structure

```
RL Based Battery Charging Optimizer/
│
├── app.py                      # Streamlit app entry point (UI + training/eval orchestration)
├── requirements.txt            # Python dependencies
│
├── artifacts/
│   └── battery_ppo_policy.pt   # Pre-trained PPO policy weights (PyTorch)
│
└── src/
    ├── envs/
    │   └── battery_env.py      # Custom battery charging environment
    │
    ├── models/
    │   └── actor_critic.py     # ActorCritic neural network (shared MLP + actor/critic heads)
    │
    ├── rl/
    │   ├── ppo_trainer.py      # PPO training loop (rollout collection + policy update)
    │   ├── rollout_buffer.py   # Rollout buffer with GAE (Generalized Advantage Estimation)
    │   └── evaluation.py       # Greedy episode evaluation
    │
    └── utils/
        └── plotting.py         # 6-panel result visualization using Matplotlib
```

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/rud12/RL-Based-Battery-Charging-Optimizer
cd "RL-Based-Battery-Charging-Optimizer"
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
- `numpy` — numerical computation
- `torch` — neural network and PPO implementation
- `matplotlib` — result visualization
- `streamlit` — interactive web dashboard


### 4. Train model

```bash
streamlit run app.py
```

1. Open `http://localhost:8501` in your browser
2. Adjust battery and training parameters in the **sidebar** as desired
3. Click **"Train model"**
4. Watch live training progress (steps, success rate, loss)
5. Once complete, the trained model is auto-saved to `artifacts/battery_ppo_policy.pt`
6. Evaluation results are displayed automatically

---

## Configuration Parameters

### Battery Parameters (Sidebar → Battery)

| Parameter | Default | Description |
|---|---|---|
| Initial Temperature | 30.0 °C | Starting battery temperature |
| Initial SoC | 20.0 % | Starting State of Charge |
| Threshold Temperature | 50.0 °C | Temperature at which episode ends as "overheat" |
| Goal SoC | 80.0 % | Target SoC to reach for a "success" |
| Max Episode Steps | 500 | Maximum timesteps per episode before "timeout" |

### Training Parameters (Sidebar → Training)

| Parameter | Default | Description |
|---|---|---|
| Total PPO Training Steps | 800,000 | Total environment interaction steps |
| Rollout Steps | 1,024 | Steps collected per PPO update |
| Learning Rate | 3e-4 | Adam optimizer learning rate |

---


## Environment Details

### Thermal Dynamics

The battery temperature evolves each timestep as:

```
ΔT = heat_generated(C-rate) - cooling_coefficient × (T - T_ambient)
```

- **Cooling coefficient**: `0.002 °C/min/°C above ambient`
- **Ambient temperature**: `25.0 °C` (fixed)
- The environment adds slight randomness at reset (±5% SoC, ±2°C) to improve policy generalization

### Episode Termination Conditions

| Condition | Result | Extra Reward |
|---|---|---|
| `temp >= threshold_temp` | **Overheat** ❌ | −250 |
| `soc >= goal_soc` | **Success** ✅ | +300 + 5 × thermal headroom |
| `steps >= max_steps` | **Timeout** ⏱️ | −20 |

---

## Reward Function

The reward at each step is shaped to encourage fast charging while penalizing thermal risk:

```python
# Reward for charging progress
reward = 10.0 * delta_soc
# Small step penalty (encourages speed)
reward -= 0.8

# C-rate bonus/penalty based on thermal headroom
if headroom > 15°C:   
    reward += 2.0 * c_rate   # Encourage high rate when cool
elif 
    headroom > 8°C:  reward += 0.8 * c_rate   # Moderate bonus
else:
    reward -= 4.0 * c_rate   # Penalize high rate near threshold

# Urgency: reward early charging
reward += 2.0 * (1 - soc_frac) * delta_soc     

# Quadratic thermal penalty when temp is within 10°C of threshold
if headroom < 10°C:
    reward -= 25.0 * ((10 - headroom) / 10)²
```

---

## Limitations

- **Simplified physics model** — the thermal and SoC dynamics are hardcoded approximations; a real battery requires electrochemical models
- **Single-environment training** — no parallel environment rollouts; training is slower than vectorized RL setups

---

## Future Improvements

- [ ] **Parallel environments** — use vectorized environments (e.g., `gym.vector`) for faster data collection
- [ ] **Realistic battery model** — integrate an equivalent-circuit or physics-based battery simulator (e.g., PyBaMM)
- [ ] **Dynamic ambient temperature** — incorporate time-varying ambient conditions as an environment input
- [ ] **Battery degradation** — model capacity fade and internal resistance increase over charge cycles
- [ ] **Multi-objective optimization** — explicitly optimize for battery health (cycle life) alongside speed and safety
---

## License

This project was developed as part of an AICTE internship. Feel free to use and adapt it for educational and research purposes.