import numpy as np

X = np.array([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0]
])

W_Q1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

W_K1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.0],
    [0.0, 0.5]
])

W_V1 = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5]
])

Q1 = np.dot(X, W_Q1)
K1 = np.dot(X, W_K1)
V1 = np.dot(X, W_V1)

print(Q1.shape)
print(K1.shape)
print(V1.shape)

K1_T = np.transpose(K1)

score1 = Q1 @ K1_T

scaled_score1 = score1 / np.sqrt(K1.shape(1))

def softmax(x):
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=1, keepdims = True)

attention_weights1 = softmax(scaled_score1)

head1_output = attention_weights1 @ V1

print("head 1 output")
print(head1_output)
print(head1_output.shape)

W_Q2 = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 0.5],
    [0.5, 0.0]
])

W_K2 = np.array([
    [0.5, 0.0],
    [0.0, 0.5],
    [0.0, 1.0],
    [1.0, 0.0]
])

W_V2 = np.array([
    [0.0, 1.0],
    [1.0, 0.0],
    [0.5, 0.5],
    [1.0, 1.0]
])

Q2 = np.dot(X, W_Q2)
K2 = np.dot(X, W_K2)
V2 = np.dot(X, W_V2)

K2_T = np.transpose(K2)

score2 = Q2 @ K2_T

scaled_score2 = score2 / np.sqrt(K1.shape[1])


attention_weights2 = softmax(scaled_score2)

head2_output = attention_weights2 @ V2

multi_head = np.concatenate([head1_output, head2_output], axis = 1)

W_O = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

output = multi_head @ W_O