import torch
import torch.nn as nn
import math


torch.manual_seed(0)

d_model = 4
num_heads = 2
d_head = 2
d_ff = 8

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.W_Q = nn.Parameter(
            torch.randn(num_heads, d_model, d_head) * 0.1
        )

        self.W_K = nn.Parameter(
            torch.randn(num_heads, d_model, d_head) * 0.1
        )
        self.W_V = nn.Parameter(
            torch.randn(num_heads, d_model, d_head) * 0.1
        )
        self.W_O = nn.Parameter(
            torch.randn(num_heads * d_head, d_model) * 0.1
        )

    def forward(self, x):

        head_outputs = []
        attention_maps = []

        for h in range(num_heads):
            Q = x @ self.W_Q[h]
            K = x @ self.W_K[h]
            V = x @ self.W_V[h]

            scores = Q @ K.transpose(-2, -1)

            scaled_scores = scores/ math.sqrt(d_head)

            attention_weights = torch.softmax(
                scaled_scores,
                dim=-1
            )

            head_output = attention_weights @ V

            head_outputs.append(head_output)
            attention_maps.append(attention_weights)

        concat = torch.cat(head_outputs, dim=-1)

        output = concat @ self.W_O

        attention_maps = torch.stack(
            attention_maps,
            dim=1
        )

        return output, attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, max_len=100, d_model=4):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                denominator = 10000**(i/d_model)

                pe[pos, i] = math.sin(pos/denominator)
                pe[pos, i+1] = math.cos(pos / denominator)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.shape[1]

        return x + self.pe[:seq_len]

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention = MultiHeadSelfAttention()

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward()

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):

        attention_output, attention_maps = self.attention(x)
        x = self.norm1(x + attention_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, attention_maps

X = torch.tensor([
    [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 1.0]
    ],
    [
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0, 0.0]
    ]
])

position = PositionalEncoding()

X = position(X)

encoder = EncoderBlock()

Y, attention_maps = encoder(X)

print("Encoder input :", X.shape)
print("Encoder output:", Y.shape)
print("Attention maps:", attention_maps.shape)

classifier = nn.Linear(d_model, 2)

pooled = Y.mean(dim=1)
logits = classifier(pooled)

print("pooled:", pooled.shape)
print("logits:", logits.shape)
print(logits)
labels = torch.tensor([1, 0])
criterion = nn.CrossEntropyLoss()

loss = criterion(logits, labels)

print("loss:", loss.item())
# 학습 전 attention 저장
with torch.no_grad():
    Y_before, attention_before = encoder(X)

WQ_before = encoder.attention.W_Q.detach().clone()

optimizer = torch.optim.SGD(
    list(encoder.parameters()) + list(classifier.parameters()),
    lr=0.1
)
with torch.no_grad():
    Y_before, _ = encoder(X)
    pooled_before = Y_before.mean(dim=1)
    logits_before = classifier(pooled_before)

    probs_before = torch.softmax(logits_before, dim=1)
    
for epoch in range(501):

    optimizer.zero_grad()

    Y, attention_maps = encoder(X)

    pooled = Y.mean(dim=1)
    logits = classifier(pooled)

    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        predictions = torch.argmax(logits, dim=1)
        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Prediction: {predictions.tolist()}"
        )

with torch.no_grad():
    Y_after, _ = encoder(X)
    pooled_after = Y_after.mean(dim=1)
    logits_after = classifier(pooled_after)

    probs_after = torch.softmax(logits_after, dim=1)

print("\n=== Prediction Before Training ===")
print("logits:")
print(logits_before)

print("probabilities:")
print(probs_before)

print("\n=== Prediction After Training ===")
print("logits:")
print(logits_after)

print("probabilities:")
print(probs_after)

print("\nlabels:", labels)