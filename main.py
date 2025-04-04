import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import imageio
from stable_baselines3 import DQN, PPO
from environment.custom_env import DentalScannerEnv

# Environment setup
env = DentalScannerEnv()
model = DQN.load("./models/ppo/ppo_final") 

# Colors and styles (same as before)
STATE_COLORS = {
    0: (0.95, 0.95, 0.95, 1.0),
    1: (0.0, 0.8, 0.0, 1.0),
    2: (0.9, 0.1, 0.1, 1.0),
    3: (0.9, 0.9, 0.0, 1.0),
    4: (0.3, 0.3, 1.0, 1.0)
}
AGENT_COLOR = (0.7, 0.0, 0.7, 1.0)

def simulate_episode():
    """Run one episode and record the agent's path"""
    obs, _ = env.reset()
    path = [tuple(np.argwhere(obs == -1)[0])]  # Start position
    
    for _ in range(100):  # Max steps
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        path.append(tuple(np.argwhere(obs == -1)[0]))
        if done:
            break
    return path

# Get actual agent path from trained model
agent_path = simulate_episode()

def draw_grid():
    # Draw background
    glColor4fv(BACKGROUND_COLOR)
    glBegin(GL_QUADS)
    glVertex2f(0, 0)
    glVertex2f(5, 0)
    glVertex2f(5, 5)
    glVertex2f(0, 5)
    glEnd()
    
    # Draw cells with colors
    for i in range(5):
        for j in range(5):
            state = state_grid[i, j]
            
            # Fill cell with state color
            glColor4fv(STATE_COLORS[state])
            glBegin(GL_QUADS)
            glVertex2f(j + 0.05, i + 0.05)
            glVertex2f(j + 0.95, i + 0.05)
            glVertex2f(j + 0.95, i + 0.95)
            glVertex2f(j + 0.05, i + 0.95)
            glEnd()
            
            # Draw cell border
            glColor4fv(BORDER_COLOR)
            glLineWidth(1.5)
            glBegin(GL_LINE_LOOP)
            glVertex2f(j + 0.05, i + 0.05)
            glVertex2f(j + 0.95, i + 0.05)
            glVertex2f(j + 0.95, i + 0.95)
            glVertex2f(j + 0.05, i + 0.95)
            glEnd()

def draw_agent(x, y):
    # Draw agent (rounded square)
    glColor4fv(AGENT_COLOR)
    glBegin(GL_QUADS)
    glVertex2f(x + 0.25, y + 0.25)
    glVertex2f(x + 0.75, y + 0.25)
    glVertex2f(x + 0.75, y + 0.75)
    glVertex2f(x + 0.25, y + 0.75)
    glEnd()
    
    # Add eyes (simple dots)
    glColor4f(1.0, 1.0, 1.0, 1.0)
    glPointSize(4.0)
    glBegin(GL_POINTS)
    glVertex2f(x + 0.35, y + 0.6)
    glVertex2f(x + 0.65, y + 0.6)
    glEnd()

def render_frame(frame_num):
    glClear(GL_COLOR_BUFFER_BIT)
    draw_grid()
    
    # Draw agent at current frame position
    if frame_num < len(agent_path):
        x, y = agent_path[frame_num]
        draw_agent(x, y)
    else:
        # Final position
        x, y = agent_path[-1]
        draw_agent(x, y)
    
    # Add step counter
    glColor3f(0, 0, 0)
    glRasterPos2f(0.1, 5.2)
    glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, ord(f'Step: {min(frame_num, len(agent_path)-1)}'))
    
    glutSwapBuffers()

def visualize_episode():
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(600, 600)
    glutCreateWindow(b"Trained Agent Visualization")
    gluOrtho2D(0, 5, 0, 5.5)
    
    glutDisplayFunc(lambda: render_frame(0))
    
    # Animation variables
    current_frame = [0]
    
    def update(value):
        current_frame[0] += 1
        if current_frame[0] <= len(agent_path) + 10:  # Extra frames at end
            glutPostRedisplay()
            glutTimerFunc(200, update, 0)  # 200ms per frame
    
    glutTimerFunc(200, update, 0)
    glutMainLoop()

if __name__ == "__main__":
    visualize_episode()