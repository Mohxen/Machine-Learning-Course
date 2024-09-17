import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Regularized Linear Regression Cost Function
def compute_cost_linear_reg(X, y, w, b, lambda_=1):
    m = X.shape[0]
    n = len(w)
    cost = 0.
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)

    reg_cost = np.sum(w**2) * (lambda_ / (2 * m))
    total_cost = cost + reg_cost
    return total_cost

# Regularized Logistic Regression Cost Function
def compute_cost_logistic_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = cost / m

    reg_cost = np.sum(w**2) * (lambda_ / (2 * m))
    total_cost = cost + reg_cost
    return total_cost

# Gradient for Linear Regression
def compute_gradient_linear_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        dj_dw += err * X[i]
        dj_db += err
    dj_dw = dj_dw / m + (lambda_ / m) * w
    dj_db = dj_db / m
    return dj_db, dj_dw

# Gradient for Logistic Regression
def compute_gradient_logistic_reg(X, y, w, b, lambda_):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        dj_dw += err_i * X[i]
        dj_db += err_i
    dj_dw = dj_dw / m + (lambda_ / m) * w
    dj_db = dj_db / m
    return dj_db, dj_dw

# Example for linear regression cost
np.random.seed(1)
X_tmp = np.random.rand(5, 6)
y_tmp = np.array([0, 1, 0, 1, 0])
w_tmp = np.random.rand(X_tmp.shape[1]) - 0.5
b_tmp = 0.5
lambda_tmp = 0.7
cost_tmp = compute_cost_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print("Regularized cost (Linear Regression):", cost_tmp)

# Example for logistic regression cost
cost_tmp = compute_cost_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print("Regularized cost (Logistic Regression):", cost_tmp)

# Example for gradients (Linear Regression)
dj_db_tmp, dj_dw_tmp = compute_gradient_linear_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"Gradient dj_db (Linear Regression): {dj_db_tmp}")
print(f"Gradient dj_dw (Linear Regression): {dj_dw_tmp}")

# Example for gradients (Logistic Regression)
dj_db_tmp, dj_dw_tmp = compute_gradient_logistic_reg(X_tmp, y_tmp, w_tmp, b_tmp, lambda_tmp)
print(f"Gradient dj_db (Logistic Regression): {dj_db_tmp}")
print(f"Gradient dj_dw (Logistic Regression): {dj_dw_tmp}")

# Example of Overfitting Visualization
# Generate a simple overfitting example
np.random.seed(42)
X = np.linspace(0, 10, 10)
y = X + np.random.randn(10) * 2  # Adding noise

# Overfitting with a high-degree polynomial
coefficients_overfit = np.polyfit(X, y, 9)  # 9th-degree polynomial
polynomial_overfit = np.poly1d(coefficients_overfit)

# A simpler linear fit
coefficients_goodfit = np.polyfit(X, y, 1)  # 1st-degree polynomial (linear)
polynomial_goodfit = np.poly1d(coefficients_goodfit)

# Plotting the results
X_test = np.linspace(0, 10, 100)
y_overfit = polynomial_overfit(X_test)
y_goodfit = polynomial_goodfit(X_test)

plt.figure()
plt.scatter(X, y, label="Data", color="blue")
plt.plot(X_test, y_overfit, label="Overfitted Model (9th-degree)", color="red")
plt.plot(X_test, y_goodfit, label="Good Fit (1st-degree)", color="green")
plt.legend()
plt.title("Overfitting Example")
plt.show()
