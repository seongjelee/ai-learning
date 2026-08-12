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

K_T = np.transpose(K)
score = Q @ K_T

print(score)
print(score.shape)

d_k = Q.shape[1]
scaled_score = score/np.sqrt(d_k)

print("scaled score")
print(scaled_score)

def softmax(x):
    exp_x = np.exp(x)

    row_sum = np.sum(exp_x, axis=1, keepdims=True)

    return exp_x/row_sum

attention_weights = softmax(scaled_score)

print("attention weights")
print(attention_weights)

print("row sums")
print(np.sum(attention_weights, axis=1))

output = attention_weights @ V

print("attention output")
print(output)

print("output shape")
print(output.shape)