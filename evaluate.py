import torch
from model.transformer import MiniTransformer

device = "cuda"
vocab_size = 5000
model = MiniTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("transformer.pt"))
model.eval()

prompt = torch.tensor([[1, 23, 45, 67]], device=device)
for _ in range(10):
    logits = model(prompt)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    prompt = torch.cat([prompt, next_token], dim=1)

print("Generated:", prompt.tolist())
