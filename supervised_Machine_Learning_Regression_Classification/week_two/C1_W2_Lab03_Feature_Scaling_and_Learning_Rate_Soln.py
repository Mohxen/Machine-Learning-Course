import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Function to generate synthetic data
def generate_synthetic_data(num_samples):
    np.random.seed(42)  # For reproducibility
    size = np.random.normal(1500, 500, num_samples)  # Size in square feet
    bedrooms = np.random.randint(1, 5, num_samples)  # Number of bedrooms
    floors = np.random.randint(1, 3, num_samples)    # Number of floors
    age = np.random.randint(0, 100, num_samples)     # Age of the house in years
    price = size * 200 + bedrooms * 50000 + floors * 20000 - age * 1000 + np.random.normal(0, 10000, num_samples)  # Price in $

    X_train = np.column_stack((size, bedrooms, floors, age))
    y_train = price / 1000  # Price in '000s
    return X_train, y_train

# Function to perform gradient descent for linear regression
def run_gradient_descent(X, y, iterations, alpha):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for i in range(iterations):
        prediction = np.dot(X, w) + b
        error = prediction - y
        gradient_w = (1/m) * np.dot(X.T, error)
        gradient_b = (1/m) * np.sum(error)
        w -= alpha * gradient_w
        b -= alpha * gradient_b
    return w, b

# Function for Z-score normalization
def zscore_normalize_features(X):
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler.mean_, scaler.scale_

# Generate synthetic dataset
X_train, y_train = generate_synthetic_data(100)
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

# Normalize the features
X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)

# Run gradient descent
w_norm, b_norm = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1)

# Predict target using normalized features
m = X_norm.shape[0]
yp = np.dot(X_norm, w_norm) + b_norm

# Visualization of data and predictions
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='Target')
    ax[i].scatter(X_train[:, i], yp, color='orange', label='Predict')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
ax[0].legend()
fig.suptitle("Target versus Prediction Using Z-Score Normalized Model")
plt.show()

# Example prediction for a specific house
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f"Predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")
