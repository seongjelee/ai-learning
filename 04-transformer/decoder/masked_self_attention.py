import torch
import torch.nn as nn
import math

torch.manual_seed(0)

d_model = 4
num_heads = 2
d_head = 2

class MaskedMultiHeadSelfAttention(nn.Module):

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

        batch_size, seq_len, _ = x.shape

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        head_outputs = []
        attention_maps = []

        for h in range(num_heads):

            Q = x @ self.W_Q[h]
            K = x @ self.W_K[h]
            V = x @ self.W_V[h]

            scores = Q @ K.transpose(-2, -1)

            scaled_scores = scores / math.sqrt(d_head)

            scaled_scores = scaled_scores.masked_fill(
                mask,
                float("-inf")
            )
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


X = torch.tensor([
    [
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 2.0, 1.0]
    ]
])

attention = MaskedMultiHeadSelfAttention()

output, attention_maps = attention(X)

print("output shape:", output.shape)
print("attention shape:", attention_maps.shape)

print("\nHead 0 attention:")
print(attention_maps[0, 0])