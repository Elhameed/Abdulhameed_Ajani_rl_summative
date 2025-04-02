import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from OpenGL.GLUT.freeglut import *
import imageio
import math

# Environment setup (same as custom_env.py)
state_grid = np.array([
    [0, 0, 0, 0, 4],  # S1 (0,0), S5 (0,4)
    [0, 2, 0, 2, 0],  # S3 at (1,1), (1,3)
    [0, 0, 2, 0, 0],  # S3 at (2,2)
    [0, 3, 0, 0, 0],  # S4 at (3,1)
    [0, 0, 0, 0, 1]   # S2 (goal) at (4,4)
])

# Enhanced color scheme with gradients (RGBA)
STATE_COLORS = {
    0: (0.95, 0.95, 0.95, 1.0),                            # Default (light gray)
    1: [(0.0, 0.7, 0.0, 1.0), (0.0, 0.9, 0.0, 1.0)],      # S2 (green gradient = success)
    2: [(0.8, 0.0, 0.0, 1.0), (1.0, 0.2, 0.2, 1.0)],      # S3 (red gradient = hazard)
    3: [(0.9, 0.9, 0.0, 1.0), (1.0, 1.0, 0.3, 1.0)],      # S4 (yellow gradient = issue)
    4: [(0.2, 0.2, 1.0, 1.0), (0.4, 0.4, 1.0, 1.0)]       # S5 (blue gradient = engagement)
}

AGENT_COLOR = (0.6, 0.0, 0.6, 1.0)  # Purple
AGENT_INNER_COLOR = (0.8, 0.3, 0.8, 1.0)  # Lighter purple
BORDER_COLOR = (0.5, 0.5, 0.5, 1.0)  # Darker borders
GRID_LINE_COLOR = (0.7, 0.7, 0.7, 1.0)  # Grid lines
TEXT_COLOR = (0.1, 0.1, 0.1, 1.0)    # Dark text
BACKGROUND_COLOR = (0.94, 0.94, 0.96, 1.0)  # Very light blue-gray

# Icons for states (simple shapes)
ICONS = {
    1: "★",  # Star for goal
    2: "⚠",  # Warning for hazard
    3: "!",   # Exclamation for issue
    4: "⟳"   # Circular arrow for engagement
}

# Simulated agent path (for animation)
agent_path = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # Moves right to S5
    (1, 4), (2, 4), (3, 4), (4, 4)           # Moves down to S2 (goal)
]

# Trail effect - tracks where agent has been
trail_opacity = 0.3
trail_cells = []

# Animation parameters
bounce_height = 0.1
rotation_angle = 0

def lerp_color(color1, color2, t):
    """Linear interpolation between two colors"""
    return tuple(a + (b - a) * t for a, b in zip(color1, color2))

def draw_grid():
    # Draw background
    glColor4fv(BACKGROUND_COLOR)
    glBegin(GL_QUADS)
    glVertex2f(0, 0)
    glVertex2f(5, 0)
    glVertex2f(5, 5)
    glVertex2f(0, 5)
    glEnd()
    
    # Draw cells with gradient colors
    for i in range(5):
        for j in range(5):
            state = state_grid[i, j]
            
            # Fill cell with state color (gradient if special state)
            if state > 0:
                # Gradient fill for special states
                colors = STATE_COLORS[state]
                glBegin(GL_QUADS)
                
                # Bottom left - color1
                glColor4fv(colors[0])
                glVertex2f(j, i)
                
                # Bottom right - interpolated
                glColor4fv(lerp_color(colors[0], colors[1], 0.5))
                glVertex2f(j + 1, i)
                
                # Top right - color2
                glColor4fv(colors[1])
                glVertex2f(j + 1, i + 1)
                
                # Top left - interpolated
                glColor4fv(lerp_color(colors[0], colors[1], 0.5))
                glVertex2f(j, i + 1)
                glEnd()
            else:
                # Solid fill for default state
                glColor4fv(STATE_COLORS[state])
                glBegin(GL_QUADS)
                glVertex2f(j, i)
                glVertex2f(j + 1, i)
                glVertex2f(j + 1, i + 1)
                glVertex2f(j, i + 1)
                glEnd()
            
            # Draw cell border with rounded corners effect
            glColor4fv(BORDER_COLOR)
            glLineWidth(2.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(j + 0.03, i + 0.03)
            glVertex2f(j + 0.97, i + 0.03)
            glVertex2f(j + 0.97, i + 0.97)
            glVertex2f(j + 0.03, i + 0.97)
            glEnd()

    # Draw grid lines
    glColor4fv(GRID_LINE_COLOR)
    glLineWidth(1.0)
    glBegin(GL_LINES)
    for i in range(6):
        glVertex2f(0, i)
        glVertex2f(5, i)
        glVertex2f(i, 0)
        glVertex2f(i, 5)
    glEnd()

def draw_agent(x, y, frame_num):
    global rotation_angle
    
    # Update rotation angle
    rotation_angle = (frame_num * 5) % 360
    
    # Calculate bounce effect
    bounce = bounce_height * abs(math.sin(frame_num * 0.2))
    
    # Draw agent shadow
    glColor4f(0.0, 0.0, 0.0, 0.2)
    glBegin(GL_QUADS)
    glVertex2f(x + 0.25, y + 0.15)
    glVertex2f(x + 0.75, y + 0.15)
    glVertex2f(x + 0.75, y + 0.25)
    glVertex2f(x + 0.25, y + 0.25)
    glEnd()
    
    # Draw agent body
    glPushMatrix()
    glTranslatef(x + 0.5, y + 0.5 + bounce, 0)
    glRotatef(rotation_angle, 0, 0, 1)
    
    # Outer shape
    glColor4fv(AGENT_COLOR)
    glBegin(GL_POLYGON)
    for i in range(8):
        angle = 2 * math.pi * i / 8
        glVertex2f(0.3 * math.cos(angle), 0.3 * math.sin(angle))
    glEnd()
    
    # Inner shape
    glColor4fv(AGENT_INNER_COLOR)
    glBegin(GL_POLYGON)
    for i in range(8):
        angle = 2 * math.pi * i / 8
        glVertex2f(0.2 * math.cos(angle), 0.2 * math.sin(angle))
    glEnd()
    
    # Eyes (black dots)
    glColor4f(0.0, 0.0, 0.0, 1.0)
    glPointSize(4.0)
    glBegin(GL_POINTS)
    glVertex2f(-0.1, 0.05)
    glVertex2f(0.1, 0.05)
    glEnd()
    
    # Smile
    glLineWidth(2.0)
    glBegin(GL_LINE_STRIP)
    for t in np.linspace(-0.1, 0.1, 5):
        glVertex2f(t, -0.05 - 0.5 * t * t)
    glEnd()
    
    glPopMatrix()

def draw_trail(frame_num):
    # Update trail cells
    if frame_num > 0 and frame_num < len(agent_path):
        if agent_path[frame_num-1] not in trail_cells:
            trail_cells.append(agent_path[frame_num-1])
    
    # Draw trail
    for idx, (tx, ty) in enumerate(trail_cells):
        # Fade trail over time
        opacity = trail_opacity * (1 - (len(trail_cells) - idx) / len(trail_cells) * 0.7)
        
        glColor4f(AGENT_COLOR[0], AGENT_COLOR[1], AGENT_COLOR[2], opacity)
        glBegin(GL_QUADS)
        glVertex2f(tx + 0.4, ty + 0.4)
        glVertex2f(tx + 0.6, ty + 0.4)
        glVertex2f(tx + 0.6, ty + 0.6)
        glVertex2f(tx + 0.4, ty + 0.6)
        glEnd()

def draw_state_icon(state, x, y):
    if state in ICONS:
        glColor4fv(TEXT_COLOR)
        glRasterPos2f(x + 0.5, y + 0.65)
        for char in ICONS[state]:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(char))

def draw_text(text, x, y, size=12):
    glColor4fv(TEXT_COLOR)
    glRasterPos2f(x + 0.5, y + 0.35)
    
    # Choose font based on size
    font = GLUT_BITMAP_HELVETICA_18 if size > 12 else GLUT_BITMAP_HELVETICA_12
    
    for char in text:
        glutBitmapCharacter(font, ord(char))

def render_frame(frame_num):
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Draw grid with state colors
    draw_grid()
    
    # Draw trail where agent has been
    draw_trail(frame_num)
    
    # Draw state icons and labels
    for i in range(5):
        for j in range(5):
            state = state_grid[i, j]
            if state > 0:
                draw_state_icon(state, j, i)
    
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
    
    # Draw title and info
    glColor4f(0.1, 0.1, 0.4, 1.0)
    glRasterPos2f(0.1, 5.2)
    title = "Dental Scanner Environment"
    for char in title:
        glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(char))
    
    # Draw frame counter
    glColor4f(0.3, 0.3, 0.3, 1.0)
    glRasterPos2f(4.2, 5.2)
    frame_text = f"Frame: {frame_num}"
    for char in frame_text:
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(char))
    
    # Animate agent
    if frame_num < len(agent_path):
        x, y = agent_path[frame_num]
        draw_agent(x, y, frame_num)
    else:
        # Keep agent at goal position
        x, y = agent_path[-1]
        draw_agent(x, y, frame_num)
    
    glutSwapBuffers()

def save_fancy_gif(filename="dental_scanner_fancy.gif", window_size=600):
    frames = []
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA)
    glutInitWindowSize(window_size, window_size)
    window = glutCreateWindow(b"Enhanced Dental Scanner")
    
    # Set up viewport with extra space for title
    glViewport(0, 0, window_size, window_size)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, 5, 0, 5.5)  # Extra space at top for title
    
    total_frames = len(agent_path) + 30  # More frames to pause at goal
    
    for frame in range(total_frames):
        # Determine which path frame to show (pause at the end)
        path_frame = min(frame, len(agent_path) - 1)
        render_frame(frame)
        
        # Capture frame
        glReadBuffer(GL_FRONT)
        pixels = glReadPixels(0, 0, window_size, window_size, GL_RGBA, GL_UNSIGNED_BYTE)
        img = np.frombuffer(pixels, dtype=np.uint8).reshape(window_size, window_size, 4)
        img = np.flipud(img)  # Flip the image vertically
        frames.append(img)
    
    # Save GIF with smoother framerate
    imageio.mimsave(filename, frames, fps=10)
    print(f"GIF saved as {filename}")
    
    # Clean up
    glutDestroyWindow(window)

if __name__ == "__main__":
    save_fancy_gif()