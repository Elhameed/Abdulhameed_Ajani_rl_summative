import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import imageio

# Environment setup
state_grid = np.array([
    [0, 0, 0, 0, 4],  # S1 (0,0), S5 (0,4)
    [0, 2, 0, 2, 0],  # S3 at (1,1), (1,3)
    [0, 0, 2, 0, 0],  # S3 at (2,2)
    [0, 3, 0, 0, 0],  # S4 at (3,1)
    [0, 0, 0, 0, 1]   # S2 (goal) at (4,4)
])

# Enhanced colors (RGBA)
STATE_COLORS = {
    0: (0.95, 0.95, 0.95, 1.0),  # Default (light gray)
    1: (0.0, 0.8, 0.0, 1.0),     # S2 (green = success)
    2: (0.9, 0.1, 0.1, 1.0),     # S3 (red = hazard)
    3: (0.9, 0.9, 0.0, 1.0),     # S4 (yellow = issue)
    4: (0.3, 0.3, 1.0, 1.0)      # S5 (blue = engagement)
}

AGENT_COLOR = (0.7, 0.0, 0.7, 1.0)  # Purple
BORDER_COLOR = (0.5, 0.5, 0.5, 1.0)  # Gray borders
TEXT_COLOR = (0.1, 0.1, 0.1, 1.0)    # Dark text
BACKGROUND_COLOR = (0.94, 0.94, 0.96, 1.0)  # Light blue-gray

# Agent path
agent_path = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # Moves right to S5
    (1, 4), (2, 4), (3, 4), (4, 4)           # Moves down to S2 (goal)
]

# Simple state icons
ICONS = {
    1: "★",  # Star for goal
    2: "⚠",  # Warning for hazard
    3: "!",  # Exclamation for issue
    4: "→"   # Arrow for engagement
}

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

def draw_text(text, x, y):
    glColor4fv(TEXT_COLOR)
    glRasterPos2f(x + 0.5, y + 0.3)
    for char in text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))

def draw_icon(icon, x, y):
    glColor4fv(TEXT_COLOR)
    glRasterPos2f(x + 0.5, y + 0.7)
    for char in icon:
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(char))

def render_frame(frame_num):
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Draw grid with state colors
    draw_grid()
    
    # Draw state icons and labels
    for i in range(5):
        for j in range(5):
            state = state_grid[i, j]
            if state > 0:
                if state in ICONS:
                    draw_icon(ICONS[state], j, i)
    
    # Label special states
    labels = {
        (4, 4): "GOAL",
        (1, 1): "HAZARD",
        (1, 3): "HAZARD",
        (2, 2): "HAZARD",
        (3, 1): "ISSUE",
        (0, 4): "ENGAGE"
    }
    
    for (i, j), text in labels.items():
        draw_text(text, j, i)
    
    # Draw title
    glColor4f(0.2, 0.2, 0.4, 1.0)
    glRasterPos2f(1.5, 5.2)
    title = "Dental Scanner"
    for char in title:
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(char))
    
    # Animate agent
    if frame_num < len(agent_path):
        x, y = agent_path[frame_num]
        draw_agent(x, y)
    else:
        # Keep agent at goal position
        x, y = agent_path[-1]
        draw_agent(x, y)
    
    glutSwapBuffers()

def save_gif(filename="dental_scanner_new.gif", window_size=500):
    frames = []
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowSize(window_size, window_size)
    window = glutCreateWindow(b"Dental Scanner")
    
    # Set up viewport with space for title
    glViewport(0, 0, window_size, window_size)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 5, 0, 5.5)  # Extra space for title
    
    total_frames = len(agent_path) + 10  # Frames to pause at goal
    
    for frame in range(total_frames):
        path_frame = min(frame, len(agent_path) - 1)
        render_frame(frame)
        
        # Capture frame
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, window_size, window_size, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(window_size, window_size, 4)
        img = np.flipud(img)  # Flip the image vertically
        frames.append(img)
    
    imageio.mimsave(filename, frames, fps=5)
    print(f"GIF saved as {filename}")
    glutDestroyWindow(window)

if __name__ == "__main__":
    save_gif()