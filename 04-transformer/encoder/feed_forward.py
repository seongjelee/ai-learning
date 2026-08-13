import numpy as np

X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 2.0, 1.0]
])

W1 = np.array([
    [ 0.2, -0.1,  0.3,  0.5, -0.4,  0.1,  0.2, -0.3],
    [ 0.4,  0.2, -0.5,  0.1,  0.3, -0.2,  0.4,  0.1],
    [-0.3,  0.5,  0.2, -0.4,  0.1,  0.6, -0.1,  0.2],
    [ 0.1, -0.2,  0.4,  0.3,  0.5, -0.3,  0.2,  0.4]
])

b1 = np.zeros(8)

def relu(x):
    return np.maximum(0, x)

W2 = np.array([
    [ 0.2,  0.1, -0.3,  0.4],
    [-0.1,  0.3,  0.2, -0.2],
    [ 0.4, -0.2,  0.1,  0.3],
    [ 0.1,  0.5, -0.4,  0.2],
    [-0.3,  0.2,  0.4,  0.1],
    [ 0.5, -0.1,  0.2, -0.3],
    [ 0.2,  0.4,  0.1, -0.2],
    [-0.2,  0.1,  0.3,  0.5]
])

b2 = np.zeros(4)

hidden = np.dot(X, W1) + b1
hidden_relu = relu(hidden)
ffn_output = np.dot(hidden_relu, W2) + b2

print("hidden shape:", hidden.shape)
print("relu output:")
print(hidden_relu)

print("FFN output shape:", ffn_output.shape)
print(ffn_output)

residual = X + ffn_output

mean = np.mean(residual, axis = 1, keepdims = True)

variance = np.var(residual, axis = 1, keepdims = True)

normalized = (residual - mean) / np.sqrt(variance + 1e-5)

gamma = np.ones(4)
beta = np.zeros(4)

encoder_output = gamma * normalized + beta

print("Encoder output:")
print(encoder_output)
print("shape:", encoder_output.shape)