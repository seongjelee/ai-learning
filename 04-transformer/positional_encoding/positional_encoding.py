import numpy as np

seq_len = 3
d_model = 4

PE = np.zeros((seq_len, d_model))

for pos in range(seq_len):
    for i in range(0, d_model, 2):
        denominator = 10000 ** (i/d_model)
        PE[pos, i] = np.sin(pos / denominator)
        PE[pos, i + 1] = np.cos(pos / denominator)

print(PE)

print("position 0:", PE[0])
print("position 1:", PE[1])
print("position 2:", PE[2])

print("0 -> 1 difference:", PE[1] - PE[0])
print("1 -> 2 difference:", PE[2] - PE[1])

X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

Z = X + PE

W_Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

W_K = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [0.0, 0.5]
])

W_V = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5]
])

Q = np.dot(Z, W_Q)
K = np.dot(Z, W_K)
V = np.dot(Z, W_V)

