import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# Simulate coffee roasting data (replace this with actual data if you have it)
def load_coffee_data():
    np.random.seed(1)
    X = np.random.uniform(175, 260, (100, 2))  # Temp between 175-260, duration between 12-15 min
    Y = np.random.randint(0, 2, (100, 1))      # Random 0 or 1 labels (good/bad roast)
    return X, Y

# Define sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Data loading
X, Y = load_coffee_data()
print(X.shape, Y.shape)

# Data normalization using TensorFlow
print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")

# Define the dense layer (manual implementation)
def my_dense(a_in, W, b):
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):               
        w = W[:,j]                                    
        z = np.dot(w, a_in) + b[j]         
        a_out[j] = sigmoid(z)  # Using the sigmoid activation function             
    return a_out

# Define sequential neural network (2-layer network)
def my_sequential(x, W1, b1, W2, b2):
    a1 = my_dense(x,  W1, b1)
    a2 = my_dense(a1, W2, b2)
    return a2

# Pre-trained weights and biases
W1_tmp = np.array([[-8.93,  0.29, 12.9], [-0.1,  -7.32, 10.81]])
b1_tmp = np.array([-9.82, -9.28,  0.96])
W2_tmp = np.array([[-31.18], [-27.59], [-32.56]])
b2_tmp = np.array([15.41])

# Define the predict function
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return p

# Test examples
X_tst = np.array([[200, 13.9], [200, 17]])   # Example for positive and negative cases
X_tstn = norm_l(X_tst)  # Normalize test examples
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

# Convert predictions to decisions (threshold = 0.5)
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")

# Simple plot of the data and decision boundaries
def plot_network(X, Y, predict_function):
    # Generate grid of points
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = norm_l(grid)  # Normalize grid points
    
    # Get predictions for the grid points
    Z = predict_function(grid_norm)
    Z = Z.reshape(xx.shape)
    
    # Plot contour and data points
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), edgecolors='k', marker='o', cmap=plt.cm.RdYlBu, s=100)
    plt.xlabel('Temperature')
    plt.ylabel('Duration')
    plt.title('Coffee Roasting Neural Network Decisions')
    plt.show()

# Plot the decision boundary and data points
plot_network(X, Y, lambda x: my_predict(x, W1_tmp, b1_tmp, W2_tmp, b2_tmp))
