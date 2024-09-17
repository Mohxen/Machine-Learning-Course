import copy
import math
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic cost function
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m
    return cost

# Plot data
def plot_data(X, y, ax):
    pos = y == 1
    neg = y == 0
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', c='r', label='Class 1')
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', c='b', label='Class 0')
    ax.legend()

# Compute gradient for logistic regression
def compute_gradient_logistic(X, y, w, b): 
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i  = f_wb_i  - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

# Perform gradient descent
def gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(compute_cost_logistic(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}")

    return w, b, J_history

# Plot logistic probabilities
def plt_prob(ax, w, b):
    x_vals = np.linspace(0, 4, 100)
    y_vals = np.linspace(0, 3.5, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = sigmoid(w[0] * X + w[1] * Y + b)
    ax.contour(X, Y, Z, levels=[0.5], colors='green')

# Example data
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Plot training data
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(X_train, y_train, ax)
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()

# Initialize weights and bias, and run gradient descent
w_tmp = np.zeros_like(X_train[0])
b_tmp = 0.
alpha = 0.1
iters = 10000
w_out, b_out, _ = gradient_descent(X_train, y_train, w_tmp, b_tmp, alpha, iters)
print(f"\nUpdated parameters: w: {w_out}, b: {b_out}")
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# Plot the probability contour (decision boundary)
plt_prob(ax, w_out, b_out)

# Plot the original data points
plot_data(X_train, y_train, ax)

# Compute and plot the decision boundary
x0 = -b_out / w_out[0]
x1 = -b_out / w_out[1]
ax.plot([0, x0], [x1, 0], c='blue', lw=1)

# Add a dummy plot for the legend to label the decision boundary
ax.plot([], [], 'g-', label='Decision Boundary')
plt.legend()  # Display the legend

ax.axis([0, 4, 0, 3.5])
plt.show()
