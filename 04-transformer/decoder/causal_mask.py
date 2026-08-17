import torch

seq_len = 4

mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
).bool()

print(mask)

scores = torch.tensor([
    [2.0, 1.0, 3.0, 2.0],
    [1.0, 4.0, 2.0, 1.0],
    [3.0, 1.0, 5.0, 2.0],
    [1.0, 2.0, 3.0, 4.0]
])

masked_scores = scores.masked_fill(
    mask,
    float("-inf")
)

print(masked_scores)

attention_weights = torch.softmax(
    masked_scores,
    dim=-1
)

print(attention_weights)