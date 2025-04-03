import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_baselines3 import DQN
from custom_env import DentalScannerEnv
import time

class AgentVisualizer:
    def __init__(self, model_path):
        self.env = DentalScannerEnv()
        self.model = DQN.load(model_path)
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.episode_history = []
        
        # Color map for states
        self.cmap = {
            0: [0.9, 0.9, 0.9],  # S1 - Light gray
            1: [0, 1, 0],        # S2 - Green (goal)
            2: [1, 0, 0],        # S3 - Red (hazard)
            3: [1, 1, 0],        # S4 - Yellow
            4: [0, 0, 1],        # S5 - Blue
            -1: [0.5, 0, 0.5]    # Agent - Purple
        }
        
    def render_grid(self):
        """Convert state grid to RGB image"""
        obs = self.env._get_obs()
        grid_rgb = np.zeros((self.env.grid_size, self.env.grid_size, 3))
        
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                state_val = obs[i, j]
                grid_rgb[i, j] = self.cmap[state_val]
        
        return grid_rgb

    def run_episode(self):
        """Run one episode and store frames"""
        obs, _ = self.env.reset()
        done = False
        frames = []
        
        while not done:
            # Get agent action
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, _, _ = self.env.step(action)
            
            # Store frame
            frames.append(self.render_grid())
            
            # Add slight delay for visualization
            time.sleep(0.2)
            
        return frames

    def animate(self, frames):
        """Create animation from frames"""
        im = self.ax.imshow(frames[0], interpolation='nearest')
        
        def update(frame):
            im.set_array(frame)
            return [im]
        
        ani = animation.FuncAnimation(
            self.fig, update, frames=frames,
            interval=200, blit=True
        )
        plt.title("Trained Agent Visualization")
        plt.axis('off')
        plt.show()

    def play(self):
        """Main play function"""
        frames = self.run_episode()
        self.animate(frames)

if __name__ == "__main__":
    visualizer = AgentVisualizer("./models/dqn/dqn_final")
    visualizer.play()