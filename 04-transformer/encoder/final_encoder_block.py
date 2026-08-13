import numpy as np

rng = np.random.default_rng(0)

seq_len = 3
d_model = 4
num_head = 2
d_head = 2
d_ff = 2

X = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 2.0, 1.0]
])

def layer_norm(x, epsilon = 1e-5):

    mean = np.mean(x, axis=1, keepdims=True)
    variance = np.var(x, axis=1, keepdims=True)

    normalized = (x - mean) / np.sqrt(variance + epsilon)

    gamma = np.ones(x.shape[1])
    beta = np.zeros(x.shape[1])

    return gamma * normalized + beta


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


W_Q = rng.normal(size=(num_head, d_model, d_head))
W_K = rng.normal(size=(num_head, d_model, d_head))
W_V = rng.normal(size=(num_head, d_model, d_head))

W_O = rng.normal(size=(num_head * d_head, d_model))

def multi_head_attention(x):
    head_outputs = []

    for h in range(num_head):
        Q = x @ W_Q[h]
        K = x @ W_K[h]
        V = x @ W_V[h]

        scores = Q @ K.T
        scaled_scores = scores / np.sqrt(d_head)

        attention_weights = softmax(scaled_scores)

        head_output = attention_weights @ V

        head_outputs.append(head_output)

    concat = np.concatenate(head_outputs, axis=1)

    output = concat @ W_O

    return output       


W1 = rng.normal(size=(d_model, d_ff))
b1 = np.zeros(d_ff)

W2 = rng.normal(size=(d_ff, d_model))
b2 = np.zeros(d_model)


def relu(x):
    return np.maximum(0, x)


def ffn(x):

    hidden = x @ W1 + b1
    hidden = relu(hidden)

    output = hidden @ W2 + b2

    return output

def encoder_block(x):

    attention_output = multi_head_attention(x)

    h = layer_norm(attention_output + x)

    ffn_output = ffn(h)

    output = layer_norm(h + ffn_output)

    return output

Y = encoder_block(X)

print("Input shape:", X.shape)
print("Encoder output:")
print(Y)
print("Encoder output shape:", Y.shape)