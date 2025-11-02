import os, torch
os.add_dll_directory(os.path.join(torch.__path__[0], 'lib'))
import bindings as tb
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

        # Fix: handle flattened (2D) or 4D outputs
        attn_out = attn_out.contiguous()
        if attn_out.dim() == 2:
            attn_out = attn_out.view(B, T, C)
        elif attn_out.dim() == 4:
            attn_out = attn_out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        else:
            raise RuntimeError(f"Unexpected attn_out shape: {attn_out.shape}")

        x = x + tb.relu(self.Wo(attn_out))
        x = x + tb.relu(self.ff2(tb.relu(self.ff1(x))))
        return x
