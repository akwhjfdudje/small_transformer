import torch
import transformer_bindings as tb
from model.attention import ScaledDotProductAttention

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.attn = ScaledDotProductAttention(self.d_k)

        self.Wq = torch.nn.Linear(d_model, d_model)
        self.Wk = torch.nn.Linear(d_model, d_model)
        self.Wv = torch.nn.Linear(d_model, d_model)
        self.Wo = torch.nn.Linear(d_model, d_model)
        self.ff1 = torch.nn.Linear(d_model, d_ff)
        self.ff2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.Wq(x).view(B, self.heads, T, self.d_k)
        K = self.Wk(x).view(B, self.heads, T, self.d_k)
        V = self.Wv(x).view(B, self.heads, T, self.d_k)

        attn_out, _ = self.attn(Q, K, V)
        attn_out = attn_out.view(B, T, C)
        x = x + tb.relu(self.Wo(attn_out))
        x = x + tb.relu(self.ff2(tb.relu(self.ff1(x))))
        return x
