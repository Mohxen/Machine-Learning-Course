import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic cost function
def logistic_cost(x, y, w, b):
    m = len(y)
    z = np.dot(x, w) + b
    predictions = sigmoid(z)
    cost = - (1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

# Plotting the data points and the logistic regression curve
def plt_simple_example(x_train, y_train, w, b):
    plt.scatter(x_train, y_train, color='red', marker='x', label='Data Points')
    
    # Generate predictions for a range of x values
    x_values = np.linspace(min(x_train), max(x_train), 100)
    y_values = sigmoid(w * x_values + b)
    
    # Plot the logistic regression curve
    plt.plot(x_values, y_values, label='Logistic Regression', color='blue')
    plt.xlabel('x')
    plt.ylabel('Probability')
    plt.title('Logistic Regression Example')
    plt.legend()
    plt.grid(True)
    plt.show()

# Squared error loss function
def squared_error_loss(x, y, w, b):
    m = len(y)
    z = np.dot(x, w) + b
    predictions = sigmoid(z)
    loss = (1/(2*m)) * np.sum((predictions - y) ** 2)
    return loss

# Plot logistic loss vs squared error loss
def plt_logistic_squared_error(x_train, y_train, w, b):
    logistic_loss = logistic_cost(x_train, y_train, w, b)
    squared_loss = squared_error_loss(x_train, y_train, w, b)
    
    plt.bar(['Logistic Loss', 'Squared Error Loss'], [logistic_loss, squared_loss], color=['blue', 'green'])
    plt.title('Comparison of Logistic Loss and Squared Error Loss')
    plt.ylabel('Loss')
    plt.show()

# Define training data
x_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.longdouble)
y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.longdouble)

# Example weights and bias (usually found through training)
w = 1.0
b = -2.5

# Visualize the simple example
plt_simple_example(x_train, y_train, w, b)

# Visualize the comparison between logistic loss and squared error loss
plt_logistic_squared_error(x_train, y_train, w, b)

# Calculate and print logistic cost
cost = logistic_cost(x_train, y_train, w, b)
print(f"Logistic Cost: {cost}")
