import torch
from model.transformer import MiniTransformer
from model.tokenizer import SPTokenizer 

vocab_size = 16000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = SPTokenizer("tokenizer")

# Load model
model = MiniTransformer(vocab_size).to(device)
model.load_state_dict(torch.load("transformer.pt"))
model.eval()

while True:

    # Encode prompt
    prompt_text = input(">> ")
    prompt = torch.tensor([tokenizer.encode(prompt_text)], device=device)

    # Generate
    max_new_tokens = 32
    for _ in range(max_new_tokens):
        logits = model(prompt)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        prompt = torch.cat([prompt, next_token], dim=1)

    # Decode output
    generated = tokenizer.decode(prompt[0].tolist())
    print()
    print("<<", generated)

