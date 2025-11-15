import os, time, torch, torch.nn as nn

# Windows DLL path fix for custom CUDA ops
if os.name == 'nt':
    os.add_dll_directory(os.path.join(torch.__path__[0], 'lib'))
else:
    os.path.join(torch.__path__[0], 'lib')

import bindings as tb
from model.transformer_block import TransformerBlock

# Baseline: pure PyTorch Transformer block
class TorchTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.ff(x)
        return x

def benchmark_forward(model, x, n=100):
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n):
            _ = model(x)
        torch.cuda.synchronize()
        end = time.time()
    return (end - start) / n

# TODO: figure out how to fix autograd torch bug
def benchmark_train(model, x, n=100):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    target = torch.randn_like(x)

    # Warm-up
    for _ in range(10):
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n):
        out = model(x)
        loss = loss_fn(out, target)
        loss.backward()
        opt.step()
        opt.zero_grad()
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / n

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, C = 8, 32, 128
    H, D_FF = 8, 512
    x = torch.randn(B, T, C, device=device)

    custom_block = TransformerBlock(C, H, D_FF).to(device)
    torch_block = TorchTransformerBlock(C, H, D_FF).to(device)

    # Correctness
    with torch.no_grad():
        out_custom = custom_block(x)
        out_torch = torch_block(x)
        diff = (out_custom - out_torch).abs().mean().item()
        print(f"Mean absolute output difference: {diff:.6f}")

    # Forward speed
    t_custom_fwd = benchmark_forward(custom_block, x)
    t_torch_fwd = benchmark_forward(torch_block, x)

    # Training speed
    t_custom_train = benchmark_train(custom_block, x)
    t_torch_train = benchmark_train(torch_block, x)

    print("\n=== TransformerBlock Benchmark ===")
    print(f"Batch size: {B}, Seq len: {T}, Hidden dim: {C}, Heads: {H}")
    print(f"Forward time: custom {t_custom_fwd*1000:.3f} ms | torch {t_torch_fwd*1000:.3f} ms | speedup {t_torch_fwd/t_custom_fwd:.2f}x")
    print(f"Train time:   custom {t_custom_train*1000:.3f} ms | torch {t_torch_train*1000:.3f} ms | speedup {t_torch_train/t_custom_train:.2f}x")
