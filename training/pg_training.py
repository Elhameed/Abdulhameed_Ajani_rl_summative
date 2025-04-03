import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from environment.custom_env import DentalScannerEnv  

# Setup directories
log_dir = "./logs/ppo/"
os.makedirs(log_dir, exist_ok=True)
model_dir = "./models/ppo/"
os.makedirs(model_dir, exist_ok=True)

# Create and wrap the environment
env = DentalScannerEnv()
env = Monitor(env, log_dir)  # Logs episode rewards, lengths, etc.

# Callbacks
checkpoint_callback = CheckpointCallback(
    save_freq=1000,  # Save model every 1000 steps
    save_path=model_dir,
    name_prefix="ppo_dental"
)

eval_callback = EvalCallback(
    env,
    eval_freq=500,  # Evaluate every 500 steps
    best_model_save_path=model_dir,
    deterministic=True,
    render=False
)

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.95,
    gae_lambda=0.95,
    ent_coef=0.01,
    tensorboard_log=log_dir,
    verbose=1
)

# Train the model
model.learn(
    total_timesteps=50_000,
    callback=[checkpoint_callback, eval_callback],
    tb_log_name="ppo_run"
)

# Save the final model
model.save(os.path.join(model_dir, "ppo_final"))

# Close the environment
env.close()