import os, torch
os.add_dll_directory(os.path.join(torch.__path__[0], 'lib'))
import bindings as tb

class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** 0.5

    def forward(self, Q, K, V):
        # Q, K, V: [batch, heads, seq_len, d_k]
        scores = tb.batched_matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = tb.softmax(scores)
        output = tb.batched_matmul(weights, V)
        return output, weights
