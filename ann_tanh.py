import numpy as np

# Inputs
x = np.array([0.05, 0.10])

# Random weights between -0.5 and 0.5
W1 = np.random.uniform(-0.5, 0.5, (2, 2))
W2 = np.random.uniform(-0.5, 0.5, (2, 2))

# Biases
b1 = 0.5
b2 = 0.7

# Activation Function
def tanh(x):
    return np.tanh(x)

# Forward Propagation
z1 = np.dot(x, W1) + b1
a1 = tanh(z1)

z2 = np.dot(a1, W2) + b2
output = tanh(z2)

print("Hidden Layer Output:")
print(a1)

print("Final Output:")
print(output)
