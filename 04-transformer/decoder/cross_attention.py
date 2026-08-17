import torch
import torch.nn as nn
import math

torch.manual_seed(0)

d_model = 4
num_heads = 2
d_head = 2


class CrossAttention(nn.Module):

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

    def forward(self, decoder_x, encoder_output):
        head_outputs = []
        attention_maps = []

        for h in range(num_heads):

            Q = decoder_x @ self.W_Q[h]

            K = encoder_output @ self.W_K[h]

            V = encoder_output @ self.W_V[h]

            scores = Q @ K.transpose(-2, -1)
            scaled_scores = scores / math.sqrt(d_head)

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

decoder_x = torch.randn(1, 2, 4)
encoder_output = torch.randn(1, 6, 4)

cross_attention = CrossAttention()

output, attention_maps = cross_attention(
    decoder_x,
    encoder_output
)

print("output:", output.shape)
print("attention:", attention_maps.shape)

print("\nHead 0:")
print(attention_maps[0, 0])
    