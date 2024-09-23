import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate x values from -2π to pπ
x = np.linspace(-2*np.pi, 2*np.pi, 1000)
# print(x)
y = np.sin(x)

# Create a DataFrame for easy handling
data = pd.DataFrame({'x': x, 'y': y})
# print(data)

# Splitting the data into training and validation
train_x = np.concatenate([data['x'].values[i:i+250] for i in range(0, 1000, 250)])
train_y = np.concatenate([data['y'].values[i:i+250] for i in range(0, 1000, 250)])
# print(f"x is {train_x} y is {train_y}")

# Randomly sample 300 points for validation
validation_indices = np.random.choice(data.index, size=300, replace=False)
validation_x = data['x'].iloc[validation_indices].values
validation_y = data['y'].iloc[validation_indices].values
# print(validation_y)

# ANN parameters
input_size = 1
hidden_size = 10
output_size = 1
learning_rate = 0.01

# Initialize weights and biases
W1 = np.random.randn(hidden_size, input_size)*0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size)*0.01
b2 = np.zeros((output_size, 1))

# Activation Function
def relu(z):
    return np.maximum(0,z)

def relu_derivative(z):
    return (z>0).astype(float)

# Forward Pass
def forward_pass(X):
    z1 = np.dot(W1, X) + b1
    a1 = relu(z1)
    z2 = np.dot(W2, a1) + b2
    return z2, a1

# Loss Function
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Backpropagation
def backpropagate(X, y_true, y_pred, a1):
    global W1, b1, W2, b2
    m = X.shape[1]

    #  Compute Gradients
    dz2 = (y_pred - y_true)/m
    dW2 = np.dot(dz2, a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)

    dz1 = np.dot(W2.T, dz2)*relu_derivative(a1)
    dW1 = np.dot(dz1, X.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    # Update weights and biases
    W1 -= learning_rate*dW1
    b1 -= learning_rate*db1
    W2 -= learning_rate*dW2
    b2 -= learning_rate*db2

# Training Loop
epochs = 1000
for epoch in range(epochs):
    # Reshape x for the forward pass
    index = np.random.randint(len(train_x))  # Randomly select an index for each training step
    X = np.array([[train_x[index]]])  # Create a 1x1 array for a single input
    y = np.array([[train_y[index]]])  # Create a 1x1 array for the corresponding output

    # Forward pass
    y_pred, a1 = forward_pass(X)

    # Compute loss
    loss = compute_loss(y, y_pred)

    # Backpropagation
    backpropagate(X, y, y_pred, a1)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, loss {loss}')

# Validation
validation_x_reshaped = validation_x.reshape(1, -1)
validation_pred, _ = forward_pass(validation_x_reshaped)

# Plot the results
plt.scatter(validation_x, validation_y, color='blue', label='Validation Data')
plt.scatter(validation_x, validation_pred.flatten(), color='red', label='ANN Predictions')
plt.title('Validation Results')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
