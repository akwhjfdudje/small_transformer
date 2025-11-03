import torch
import torch.nn as nn
import torch.optim as optim
from model.transformer import MiniTransformer

# Configuration 
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 128       #simple byte-level vocab
seq_len = 32
batch_size = 16
epochs = 3000
lr = 3e-4

# Prepare a simple corpus 
corpus = [
    "to be or not to be",
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "a stitch in time saves nine",
    "an apple a day keeps the doctor away",
    "all that glitters is not gold",
    "practice makes perfect",
    "knowledge is power",
    "time waits for no one",
    "better late than never"
]

# Convert chars to IDs (simple ASCII tokenizer)
text = "\n".join(corpus)
data = torch.tensor([ord(c) % vocab_size for c in text], dtype=torch.long)

def get_batch():
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

# Model 
model = MiniTransformer(vocab_size, d_model=128, num_layers=2, heads=4, d_ff=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Training 
for epoch in range(epochs):
    x, y = get_batch()
    logits = model(x)
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Save the model 
torch.save(model.state_dict(), "transformer.pt")
print("Saved: transformer.pt")

