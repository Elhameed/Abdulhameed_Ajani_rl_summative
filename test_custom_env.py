from environment.custom_env import DentalScannerEnv

def run_tests():
    print("=== Testing DentalScannerEnv ===")
    env = DentalScannerEnv()
    
    # Test 1: Path to goal (S2)
    print("\nTest 1: Optimal path to goal (Right x4, Down x4)")
    obs, _ = env.reset()
    env.render()
    for action in [3, 3, 3, 3, 1, 1, 1, 1]:  # Right, Down
        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated:
            print(f"SUCCESS! Reached goal. Reward: {reward}")
            break
        if truncated:
            print("FAILED! Episode truncated.")

    # Test 2: Invalid actions
    print("\nTest 2: Invalid actions (Up from start)")
    obs, _ = env.reset()
    for _ in range(5):
        _, reward, terminated, truncated, _ = env.step(0)  # Up
        env.render()
        if truncated:
            print(f"PASS! Episode truncated after invalid actions. Final reward: {reward}")
            break

    # Test 3: Hazard retry limit
    print("\nTest 3: Hazard retry limit (3 hits)")
    obs, _ = env.reset()
    for action in [1, 3, 1, 2, 1, 3, 1]:  # Hits (1,1), (1,3), (2,2)
        _, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated:
            print(f"TERMINATED at retry {env.retry_count}. Final reward: {reward}")
            break
        if truncated:
            print("TRUNCATED (max steps)")

    # Test 4: Pure max steps test
    print("\nTest 4: Pure Max Steps Test")
    obs, _ = env.reset()
    action_sequence = [3, 1, 2, 1, 3, 0, 2]  # Safe path
    for i in range(env.max_steps + 1):
        action = action_sequence[i % len(action_sequence)]
        _, reward, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            status = "TERMINATED" if terminated else "TRUNCATED"
            print(f"{status} after {env.step_count} steps. Reward: {reward}")
            break

if __name__ == "__main__":
    run_tests()