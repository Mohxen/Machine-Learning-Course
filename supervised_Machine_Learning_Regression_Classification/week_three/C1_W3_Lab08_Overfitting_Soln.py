import numpy as np
import matplotlib.pyplot as plt

# Generate some random data with noise
np.random.seed(42)
X = np.random.rand(20, 1) * 10
y = 2 * X + 1 + np.random.randn(20, 1) * 2  # Linear trend with noise

# Fit a high-degree polynomial (overfitting example)
p_overfit = np.poly1d(np.polyfit(X.flatten(), y.flatten(), 15))

# Fit a linear model (good fit)
p_goodfit = np.poly1d(np.polyfit(X.flatten(), y.flatten(), 1))

# Generate values for plotting
X_line = np.linspace(0, 10, 100)

# Plot the data and the two fits
plt.scatter(X, y, label='Data', color='blue')
plt.plot(X_line, p_overfit(X_line), label='Overfitted Model', color='red')
plt.plot(X_line, p_goodfit(X_line), label='Good Fit', color='green')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Overfitting Example")
plt.show()
