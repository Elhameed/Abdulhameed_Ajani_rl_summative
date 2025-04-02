import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

# Define states (mapped to grid cells)
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
        self.action_space = spaces.Discrete(4)  # [0: Up, 1: Down, 2: Left, 3: Right]
        self.observation_space = spaces.Box(
            low=0, high=len(State), 
            shape=(self.grid_size, self.grid_size), 
            dtype=np.int32
        )
        
        # State-to-cell mapping
        self.state_grid = np.array([
            [State.FIRST_TIME.value, 0, 0, 0, State.USER_ENGAGED.value],
            [0, State.POOR_QUALITY.value, 0, State.POOR_QUALITY.value, 0],
            [0, 0, State.POOR_QUALITY.value, 0, 0],
            [0, State.ISSUE_DETECTED.value, 0, 0, 0],
            [0, 0, 0, 0, State.SUCCESS_SCAN.value]
        ])
        
        self.agent_pos = [0, 0]  # Start at S1 (top-left)
        self.retry_count = 0
        self.max_retries = 3
        self.invalid_actions = 0
        self.max_invalid_actions = 3

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.retry_count = 0
        self.invalid_actions = 0
        return self._get_obs(), {}

    def step(self, action):
        new_pos = self.agent_pos.copy()
        
        # Movement actions
        if action == 0: new_pos[0] -= 1  # Up
        elif action == 1: new_pos[0] += 1  # Down
        elif action == 2: new_pos[1] -= 1  # Left
        elif action == 3: new_pos[1] += 1  # Right
        
        # Check boundaries
        if (0 <= new_pos[0] < self.grid_size) and (0 <= new_pos[1] < self.grid_size):
            self.agent_pos = new_pos
            invalid_action = False
        else:
            invalid_action = True

        # Get current state
        current_state = self.state_grid[self.agent_pos[0], self.agent_pos[1]]
        terminated = False
        truncated = False
        reward = 0

        # State-specific logic
        if current_state == State.POOR_QUALITY.value:  # S3
            reward = -5
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                terminated = True  # Exceeded retry limit

        elif current_state == State.SUCCESS_SCAN.value:  # S2
            reward = 10
            terminated = True  # Success!

        elif current_state == State.ISSUE_DETECTED.value:  # S4
            reward = 2  # Small reward for reaching S4

        elif current_state == State.USER_ENGAGED.value:  # S5
            reward = 3  # Bonus for engagement

        elif invalid_action:
            reward = -1
            self.invalid_actions += 1
            if self.invalid_actions >= self.max_invalid_actions:
                truncated = True  # Too many invalid actions

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        """Return agent's current state (grid with agent position)."""
        obs = self.state_grid.copy()
        obs[self.agent_pos[0], self.agent_pos[1]] = -1  # Mark agent
        return obs

    def render(self, mode='human'):
        """Simple console rendering (PyOpenGL will be separate)."""
        grid = np.array(self.state_grid, dtype=str)
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'  # Agent
        print(grid)