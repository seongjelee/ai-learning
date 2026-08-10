import numpy as np

X = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

W_Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

W_K = np.array([
    [0.5, 0.5],
    [0.5, -0.5]
])

W_V = np.array([
    [1.0, 1.0],
    [0.0, 1.0]
])

Q = np.dot(X, W_Q)
K = np.dot(X, W_K)
V = np.dot(X, W_V)

print("Q")
print(Q)

print("K")
print(K)

print("V")
print(V)