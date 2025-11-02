import torch
import transformer_bindings as tb

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** 0.5

    def forward(self, Q, K, V):
        # Q, K, V: [batch, heads, seq_len, d_k]
        scores = tb.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = tb.softmax(scores)
        output = tb.matmul(weights, V)
        return output, weights
