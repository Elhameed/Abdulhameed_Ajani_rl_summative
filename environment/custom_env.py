import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

class State(Enum):
    FIRST_TIME = 0    # S1: Start position (0,0)
    SUCCESS_SCAN = 1  # S2: Goal position (4,4)
    POOR_QUALITY = 2  # S3: Hazard cells
    ISSUE_DETECTED = 3  # S4: Intermediate cells
    USER_ENGAGED = 4   # S5: Bonus cells

class DentalScannerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.grid_size = 5
        self.action_space = spaces.Discrete(4)  # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = spaces.Box(
            low=-1, 
            high=4,  # -1 for agent, 0-4 for states
            shape=(self.grid_size, self.grid_size),
            dtype=np.int32
        )
        
        # State grid layout
        self.state_grid = np.array([
            [State.FIRST_TIME.value, 0, 0, 0, State.USER_ENGAGED.value],
            [0, State.POOR_QUALITY.value, 0, State.POOR_QUALITY.value, 0],
            [0, 0, State.POOR_QUALITY.value, 0, 0],
            [0, State.ISSUE_DETECTED.value, 0, 0, 0],
            [0, 0, 0, 0, State.SUCCESS_SCAN.value]
        ])
        
        # Agent tracking
        self.agent_pos = [0, 0]
        self.step_count = 0
        self.retry_count = 0
        self.invalid_actions = 0
        self.last_reward = 0
        
        # Limits
        self.max_steps = 100
        self.max_retries = 3
        self.max_invalid_actions = 3

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.step_count = 0
        self.retry_count = 0
        self.invalid_actions = 0
        self.last_reward = 0
        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        new_pos = self.agent_pos.copy()
        
        # Movement logic
        if action == 0: new_pos[0] -= 1  # Up
        elif action == 1: new_pos[0] += 1  # Down
        elif action == 2: new_pos[1] -= 1  # Left
        elif action == 3: new_pos[1] += 1  # Right
        
        # Check boundaries
        invalid_action = not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size)
        
        if not invalid_action:
            self.agent_pos = new_pos
            current_state = self.state_grid[self.agent_pos[0], self.agent_pos[1]]
        else:
            current_state = None

        # Initialize return values
        reward = 0
        terminated = False
        truncated = (self.step_count >= self.max_steps)
        
        # State-specific logic
        if invalid_action:
            reward = -1
            self.invalid_actions += 1
            if self.invalid_actions >= self.max_invalid_actions:
                truncated = True
        elif current_state == State.POOR_QUALITY.value:  # S3
            reward = -5
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                terminated = True
                reward = -10  # Additional penalty for max retries
        elif current_state == State.SUCCESS_SCAN.value:  # S2
            reward = 10
            terminated = True
        elif current_state == State.ISSUE_DETECTED.value:  # S4
            reward = 2
        elif current_state == State.USER_ENGAGED.value:  # S5
            reward = 3
        else:  # Unmarked cell
            reward = 0

        self.last_reward = reward
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Return grid with agent marked as -1."""
        obs = self.state_grid.copy()
        obs[self.agent_pos[0], self.agent_pos[1]] = -1
        return obs

    def render(self, mode='human'):
        """Enhanced debugging output"""
        grid = np.array(self.state_grid, dtype=str)
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'
        print(f"Step {self.step_count}: Reward={self.last_reward}")
        print(grid)
        print(f"Pos: {self.agent_pos}, Retries: {self.retry_count}/{self.max_retries}, Invalid: {self.invalid_actions}/{self.max_invalid_actions}")
        print("-----")