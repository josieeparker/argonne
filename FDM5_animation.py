# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:12:12 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
from matplotlib.animation import ArtistAnimation
# we don't actually need all of these but its from me playing around to get a gif


def laplace_operator(u, dx, dy):
    laplace_u = np.zeros_like(u)
    laplace_u[1:-1, 1:-1] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    return laplace_u

def solve_system(u, w, dx, dy, dt, max_time_steps=1000, tol=1e-5):
    u_history = [u.copy()]  # Store the initial condition
    w_history = [w.copy()]  # Store the initial condition

    for time_step in range(1, max_time_steps + 1):
        # Update u using the given solution 
        u_new = np.exp(-4 * (X - 0.5)**2 - 4 * (Y - 0.5)**2)

        # Update w using Neumann boundary conditions
        w_new = w.copy()
        w_new[1:-1, 0] = w[1:-1, 1]
        w_new[1:-1, -1] = w[1:-1, -2]
        w_new[0, :] = w[1, :]
        w_new[-1, :] = w[-2, :]

        # Laplace operator for the interior points
        laplace_w = laplace_operator(w, dx, dy)

        # Update w using the Poisson equation with explicit Euler scheme
        w_new[1:-1, 1:-1] = w[1:-1, 1:-1] - dt * laplace_w[1:-1, 1:-1]

        # Check for convergence
        u_diff_norm = np.linalg.norm(u_new - u)
        w_diff_norm = np.linalg.norm(w_new - w)

        # Store the results for plotting
        u_history.append(u_new.copy())
        w_history.append(w_new.copy())

        # Check for convergence
        if u_diff_norm < tol and w_diff_norm < tol:
            break

        # Update for the next time step
        u = u_new
        w = w_new

    return u_history, w_history

# Define parameters and grid
nx, ny = 50, 50
Lx, Ly = 1.0, 1.0
dx, dy = Lx / (nx - 1), Ly / (ny - 1)

# Initialize u and w
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

u = np.exp(-4 * (X - 0.5)**2 - 4 * (Y - 0.5)**2)  # Initial guess for u
w = np.zeros((nx, ny))  # Initial guess for w

# Set other parameters
max_time_steps = 1000
dt = 0.01  # Time step size
tolerance = 1e-5

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3)

# Initialize empty images
img1 = ax1.imshow(u, cmap='viridis', extent=(0, Lx, 0, Ly), origin='lower', animated=True)
img2 = ax2.imshow(w, cmap='viridis', extent=(0, Lx, 0, Ly), origin='lower', animated=True)

# Set axis labels and titles
ax1.set_xlabel('x')
ax1.set_ylabel('y')

ax2.set_xlabel('x')
ax2.set_ylabel('y')

def update(frame):
    img1.set_array(u_history[frame])
    img2.set_array(w_history[frame])

    # Display current time step and tolerance as text annotations
    ax1.text(0.02, 0.92, f'Time Step: {frame}\nTolerance: {tolerance:.5e}', transform=ax1.transAxes, ha='left', va='top', color='white')
    ax2.text(0.02, 0.92, f'Time Step: {frame}\nTolerance: {tolerance:.5e}', transform=ax2.transAxes, ha='left', va='top', color='white')

    # Update axis labels and titles
    ax1.set_title(f'Solution for u (Time Step {frame}) - Tolerance: {tolerance:.5e}')
    ax2.set_title(f'Solution for w (Time Step {frame}) - Tolerance: {tolerance:.5e}')

    return img1, img2

# Solve the system over multiple time steps
u_history, w_history = solve_system(u, w, dx, dy, dt, max_time_steps, tol=tolerance)

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3)

# Initialize empty images
img1 = ax1.imshow(u_history[0], cmap='viridis', extent=(0, Lx, 0, Ly), origin='lower')
img2 = ax2.imshow(w_history[0], cmap='viridis', extent=(0, Lx, 0, Ly), origin='lower')

# Set axis labels and titles
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

# Create the animation
animation = FuncAnimation(fig, update, frames=len(u_history), interval=10, repeat=False)

# Save the animation as a gif file
animation.save('solution_animation.gif', writer='pillow', fps=20)

# Show the animation
plt.show()
