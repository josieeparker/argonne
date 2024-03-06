# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 12:22:51 2024

@author: User
"""
# pinn_solver.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()

        # Define the neural network layers for u
        self.u_nn = self.build_nn()

        # Define the neural network layers for w
        self.w_nn = self.build_nn()

    def build_nn(self):
        model = tf.keras.Sequential([
            layers.InputLayer(input_shape=(2,)),  # Input: (x, y)
            layers.Dense(50, activation='tanh'),
            layers.Dense(50, activation='tanh'),
            layers.Dense(1, activation=None)  # Output: u or w
        ])
        return model

    def neural_network(self, inputs, network='u'):
        if network == 'u':
            return self.u_nn(inputs)
        elif network == 'w':
            return self.w_nn(inputs)
        else:
            raise ValueError("Invalid network type. Use 'u' or 'w'.")

    def laplace_operator(self, inputs, network='u'):
        with tf.GradientTape(persistent=True) as laplace_tape:
            laplace_tape.watch(inputs)
            u = self.neural_network(inputs, network=network)

        laplace_u_x = laplace_tape.gradient(laplace_u, inputs)[:, 0]
        laplace_u_y = laplace_tape.gradient(laplace_u, inputs)[:, 1]

        del laplace_tape

        return laplace_u_x, laplace_u_y

    def train_step(self, x_data, y_data):
        with tf.GradientTape(persistent=True) as tape:
            u_pred = self.neural_network(tf.stack([x_data, y_data], axis=1), network='u')
            w_pred = self.neural_network(tf.stack([x_data, y_data], axis=1), network='w')

            # Define the PDE residual terms
            laplace_u_x, laplace_u_y = self.laplace_operator(tf.stack([x_data, y_data], axis=1), network='u')
            pde_residual_u = laplace_u_x + laplace_u_y  # Example PDE term (customize based on your PDE)

            # Compute PINN loss
            loss = tf.reduce_mean(tf.square(u_pred - u_true(x_data, y_data))) + \
                   tf.reduce_mean(tf.square(w_pred - w_true(x_data, y_data))) + \
                   tf.reduce_mean(tf.square(pde_residual_u))

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        del tape

        return loss

# Function to generate true solution for u
def u_true(x, y):
    return np.exp(-4 * (x - 0.5)**2 - 4 * (y - 0.5)**2)

# Function to generate true solution for w
def w_true(x, y):
    return u_true(x, y)**2  # Example relationship (customize based on your PDE)
