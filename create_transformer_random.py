import torch
from model.transformer import MiniTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 128 

model = MiniTransformer(vocab_size).to(device)

# Use a stable, deterministic initialization
for name, param in model.named_parameters():
    if "weight" in name:
        torch.nn.init.xavier_uniform_(param)
    elif "bias" in name:
        torch.nn.init.zeros_(param)

torch.save(model.state_dict(), "transformer.pt")
print("Saved transformer.pt")
