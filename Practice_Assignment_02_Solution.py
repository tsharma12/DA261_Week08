# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 17:05:41 2025

@author: Teena Sharma
"""


"""# Question 1
"""

import numpy as np
import matplotlib.pyplot as plt

def compute_y_values(a, b, x_values):
    return a*x_values + b

def compute_elementwise_error(a, b, x_values, y_values):
    y_hat = compute_y_values(a, b, x_values)
    return y_values - y_hat

def compute_average_error(errors):
    return np.mean(errors ** 2)

def gradient_descent(a, b, x_values, y_values, learning_rate, iterations):
    errors_history = []
    a_values = [a]
    b_values = [b]

    for iter in range(iterations):
        errors = compute_elementwise_error(a, b, x_values, y_values)
        errors_history.append(compute_average_error(errors))

        gradient_a = -2 * np.mean(errors * x_values)
        gradient_b = -2 * np.mean(errors)

        a -= learning_rate * gradient_a
        b -= learning_rate * gradient_b

        a_values.append(a)
        b_values.append(b)

    return a, b, a_values, b_values, errors_history

# Parameters
a_initial = 1.0
b_initial = 1.0
learning_rates = [0.001, 0.01]  # Vary the learning rate
iterations = 2000  # Increased the number of iterations for better convergence
n = 200

# Generate x values
x_values = np.linspace(-15, 15, n)

# Generate noisy y values
y_values = -5*x_values + 7 + np.random.normal(0, 1, n)

# Compute solutions using Gradient Descent for different learning rates
solutions = {}
for learning_rate in learning_rates:
    a, b, a_values, b_values, errors_history = gradient_descent(a_initial, b_initial, x_values, y_values, learning_rate, iterations)
    solutions[learning_rate] = {
                                'a_values': a_values,
                                'b_values': b_values,
                                }

# Contour plot of the error function
a_vals = np.linspace(-20, 20, 200)
b_vals = np.linspace(-20, 20, 200)
A, B = np.meshgrid(a_vals, b_vals)
error_values = np.zeros_like(A)

for i in range(len(a_vals)):
    for j in range(len(b_vals)):
        errors = compute_elementwise_error(A[i, j], B[i, j], x_values, y_values)
        error_values[i, j] = compute_average_error(errors)

# Plot the contour plot of the error function
plt.figure(figsize=(8, 6))
plt.contour(A, B, error_values, levels=50, cmap='viridis')
plt.colorbar(label='Average Error (E)')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Contour Plot of Error Function')

# Plot the trajectory of solutions for different learning rates
for learning_rate in learning_rates:
    plt.plot(solutions[learning_rate]['a_values'], solutions[learning_rate]['b_values'], '-o', label=f'LR={learning_rate}')

plt.legend()
plt.show()


