import torch
import torch.nn.functional as F
from model.transformer import MiniTransformer

device = "cuda"
vocab_size = 5000
model = MiniTransformer(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Dummy dataset
data = torch.randint(0, vocab_size, (64, 32), device=device)
target = torch.randint(0, vocab_size, (64, 32), device=device)

for epoch in range(5):
    optimizer.zero_grad()
    logits = model(data)
    loss = F.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: loss = {loss.item():.4f}")
