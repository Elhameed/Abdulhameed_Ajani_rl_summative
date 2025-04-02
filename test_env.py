from environment.custom_env import DentalScannerEnv

# Initialize the environment
env = DentalScannerEnv()
obs, info = env.reset()
print("Initial Observation:")
env.render()  # Show the grid with agent position

# Test hazard retry limit
actions = [3, 1, 1, 1]  # Right, Down (S3), Down (invalid), Down (S3 again)
for action in actions:
    print(f"\nAction: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    if terminated or truncated:
        print("Episode ended!")
        break