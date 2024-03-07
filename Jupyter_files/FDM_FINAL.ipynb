# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:05:37 2024

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from pinn_solver import PINN, u_true, w_true
import tensorflow as tf 

tf.config.run_functions_eagerly(True)

def laplace_operator(u, dx, dy):
    laplace_u = np.zeros_like(u)
    laplace_u[1:-1, 1:-1] = (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2 + \
                           (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
    return laplace_u

def solve_system_with_pinn(pinn, u, w, dx, dy, max_iter=10000, tol=1e-5):
    for iteration in range(max_iter):
        # Update u using the given solution form
        u_new = np.exp(-4 * (X - 0.5)**2 - 4 * (Y - 0.5)**2)

        # Update w using Neumann boundary conditions
        w_new = w.copy()

        # Neumann boundary conditions at x = 0, Lx
        w_new[1:-1, 0] = w[1:-1, 1]
        w_new[1:-1, -1] = w[1:-1, -2]

        # Neumann boundary conditions at y = 0, Ly
        w_new[0, :] = w[1, :]
        w_new[-1, :] = w[-2, :]

        # Laplace operator for the interior points
        laplace_w = laplace_operator(w, dx, dy)

        # Update w using the Poisson equation
        w_new[1:-1, 1:-1] = w[1:-1, 1:-1] - laplace_w[1:-1, 1:-1]

        # Generate random data points for training
        x_data = np.random.uniform(0, 1, size=(100, 1))
        y_data = np.random.uniform(0, 1, size=(100, 1))

        # Convert data to TensorFlow tensors
        x_data_tf = tf.convert_to_tensor(x_data, dtype=tf.float32)
        y_data_tf = tf.convert_to_tensor(y_data, dtype=tf.float32)

        # Training step with PINN
        loss_pinn = pinn.train_step(x_data_tf, y_data_tf)

        # Print and check for convergence
        print(f'Iteration {iteration}, PINN loss: {loss_pinn.numpy()}')

        # Check for convergence and print intermediate results
        u_diff_norm = np.linalg.norm(u_new - u)
        w_diff_norm = np.linalg.norm(w_new - w)
        print(f'Iteration {iteration}, u norm: {u_diff_norm}, w norm: {w_diff_norm}')

        # Check for convergence
        if u_diff_norm < tol and w_diff_norm < tol:
            break

        # Update for the next iteration
        u = u_new
        w = w_new

    return u_new, w_new


if __name__ == '__main__':
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
    max_iterations = 1000
    tolerance = 1e-5

    # Create PINN instance
    pinn = PINN()

    # Solve the system
    u_sol, w_sol = solve_system_with_pinn(pinn, u, w, dx, dy, max_iter=max_iterations, tol=tolerance)

    # Plot the results
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, u_sol.T, cmap='viridis', levels=20)
    plt.title('Solution for u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, w_sol.T, cmap='viridis', levels=20)
    plt.title('Solution for w')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
