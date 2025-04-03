import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from environment.custom_env import DentalScannerEnv 

# Setup directories
log_dir = "./logs/dqn/"
os.makedirs(log_dir, exist_ok=True)
model_dir = "./models/dqn/"
os.makedirs(model_dir, exist_ok=True)

# Create and wrap the environment
env = DentalScannerEnv()
env = Monitor(env, log_dir)  # Logs episode rewards, lengths, etc.

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # Save model every 1000 steps
    save_path=model_dir,
    name_prefix="dqn_dental"
)

eval_callback = EvalCallback(
    env,
    eval_freq=500,  # Evaluate every 500 steps
    best_model_save_path=model_dir,
    n_eval_episodes=5,
    deterministic=True,
    warn=False,
    render=False
)

# Initialize DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    batch_size=64,
    gamma=0.99,
    exploration_final_eps=0.1,
    exploration_fraction=0.2,
    tensorboard_log=log_dir,
    verbose=1
)

# Train the model
model.learn(
    total_timesteps=50_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="dqn_run"
)

# Save the final model
model.save(os.path.join(model_dir, "dqn_final"))

# Close the environment
env.close()