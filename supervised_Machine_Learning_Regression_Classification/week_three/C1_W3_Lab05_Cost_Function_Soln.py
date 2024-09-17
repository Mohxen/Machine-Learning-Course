import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to plot the data
def plot_data(X, y, ax):
    # Separate the classes
    pos = y == 1  # class 1
    neg = y == 0  # class 0

    # Plot the data points
    ax.scatter(X[pos, 0], X[pos, 1], marker='o', color='blue', label='Class 1')
    ax.scatter(X[neg, 0], X[neg, 1], marker='x', color='red', label='Class 0')

# Define the logistic cost function
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log(1 - f_wb_i)
    cost = cost / m
    return cost

# Data
X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

# Plot the data
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend()
plt.show()

# Compute the logistic cost for given parameters
w_tmp = np.array([1, 1])
b_tmp = -3
print(f"Cost for w={w_tmp}, b={b_tmp}: {compute_cost_logistic(X_train, y_train, w_tmp, b_tmp)}")

# Plot decision boundaries
x0 = np.arange(0, 6)  # Values between 0 and 6 for x_0
x1 = 3 - x0  # First decision boundary (for b=-3)
x1_other = 4 - x0  # Second decision boundary (for b=-4)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

# Plot the decision boundary
ax.plot(x0, x1, color='blue', label="$b$=-3")
ax.plot(x0, x1_other, color='magenta', label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train, y_train, ax)
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()

# Compute and print costs for both decision boundaries
w_array1 = np.array([1, 1])
b_1 = -3
w_array2 = np.array([1, 1])
b_2 = -4

print(f"Cost for b = -3: {compute_cost_logistic(X_train, y_train, w_array1, b_1)}")
print(f"Cost for b = -4: {compute_cost_logistic(X_train, y_train, w_array2, b_2)}")
