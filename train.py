import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from model.transformer import MiniTransformer
from model.tokenizer import SimpleTokenizer

# Config 
device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
SEQ_LEN = 16
D_MODEL = 128
HEADS = 4
D_FF = 256
LAYERS = 2
EPOCHS = 5
LR = 3e-4

# Load or create dataset 
with open("data/gutenberg/1661.txt", "r", encoding="utf-8") as f:
    corpus = [line.strip() for line in f if line.strip()]

# Tokenizer 
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(corpus)
vocab_size = len(tokenizer.vocab)
print(f"Vocab size: {vocab_size}")

encoded = [torch.tensor(tokenizer.encode(text)) for text in corpus]
data = torch.cat(encoded)
print(f"Total tokens: {len(data)}")

# Dataset 
class WordDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y

dataset = WordDataset(data, SEQ_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model 
model = MiniTransformer(vocab_size, D_MODEL, LAYERS, HEADS, D_FF).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# Training loop 
for epoch in range(EPOCHS):
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss / len(loader):.4f}")

# Save model + vocab 
torch.save(model.state_dict(), "transformer.pt")
torch.save(tokenizer.vocab, "vocab.pt")
print("Saved transformer.pt and vocab.pt")
