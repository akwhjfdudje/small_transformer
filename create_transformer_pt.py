import torch
from model.transformer import MiniTransformer

# --- Config ---
VOCAB_SIZE = 5000
D_MODEL = 128
NUM_LAYERS = 2
HEADS = 4
D_FF = 256
EPOCHS = 100
LR = 1e-3
SEQ_LEN = 32
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Initialize model ---
model = MiniTransformer(VOCAB_SIZE, D_MODEL, NUM_LAYERS, HEADS, D_FF).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# --- Tiny synthetic dataset ---
def random_batch():
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    y = torch.roll(x, shifts=-1, dims=1)
    return x, y

print("Training a small transformer model (synthetic)...")
model.train()
for epoch in range(EPOCHS):
    x, y = random_batch()
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# --- Save the trained model ---
torch.save(model.state_dict(), "transformer.pt")
print("\n✅ Saved transformer.pt successfully!")

