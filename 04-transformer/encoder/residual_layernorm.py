import numpy as np

x = np.array([
    [2.0, 4.0, 6.0, 8.0],
    [1.0, 2.0, 3.0, 4.0],
    [10.0, 20.0, 30.0, 40.0]
])

mean = np.mean(x, axis=1, keepdims=True)
variance = np.var(x, axis=1, keepdims=True)
epsilon = 1e-5

x_normalized = (x-mean) / (np.sqrt(variance+epsilon))
print(x_normalized)

gamma = np.ones(4)
beta = np.zeros(4)

output = gamma * x_normalized + beta