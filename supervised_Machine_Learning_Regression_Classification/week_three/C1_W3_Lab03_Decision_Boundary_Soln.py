import numpy as np
import matplotlib.pyplot as plt

# Sample data and labels
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)

# Function to plot data
def plot_data(X, y, ax):
    pos = y.flatten() == 1
    neg = y.flatten() == 0
    ax.scatter(X[pos, 0], X[pos, 1], marker='x', color='red', label="Class 1")
    ax.scatter(X[neg, 0], X[neg, 1], marker='o', color='blue', label="Class 0")
    ax.legend()

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Function to draw a vertical threshold line (example at x=0)
def draw_vthresh(ax, thresh):
    ax.axvline(x=thresh, color='red', linestyle='--')

# Plot data
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()

# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10, 11)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax, 0)
plt.show()

# Choose values between 0 and 6
x0 = np.arange(0, 6)
x1 = 3 - x0

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
# Plot the decision boundary
ax.plot(x0, x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0, x1, alpha=0.2)

# Plot the original data
plot_data(X, y, ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
p