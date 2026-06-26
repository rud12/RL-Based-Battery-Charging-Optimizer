# PPO Battery Charging Optimizer

This project is a modular version of the original single-file PPO battery charging code. It keeps the same battery environment, PPO actor-critic model, rollout buffer, trainer, evaluation logic, and plotting, with a simple Streamlit interface.

## Structure

- `src/envs/battery_env.py` - battery charging environment
- `src/models/actor_critic.py` - PPO actor-critic network
- `src/rl/rollout_buffer.py` - rollout storage and GAE returns
- `src/rl/ppo_trainer.py` - PPO training loop
- `src/rl/evaluation.py` - policy evaluation and summary formatting
- `src/utils/plotting.py` - Matplotlib result plots
- `app.py` - Streamlit app

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app can train a model and save it to `artifacts/battery_ppo_policy.pt`, or load that saved model and run evaluation.

