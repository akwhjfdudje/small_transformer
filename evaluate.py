import torch
from model.transformer import MiniTransformer

# Configuration 
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 128  

# Load model 
model = MiniTransformer(vocab_size, d_model=128, num_layers=2, heads=4, d_ff=256).to(device)
model.load_state_dict(torch.load("transformer.pt", map_location=device))
model.eval()

while True:
    # Encode prompt as ASCII values 
    prompt_text = input(">> ")
    prompt = torch.tensor([[ord(c) % vocab_size for c in prompt_text]], device=device)

    # Generate new characters 
    gen_len = 100
    for _ in range(gen_len):
        logits = model(prompt)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        next_token = next_token.clamp(0, vocab_size - 1)
        prompt = torch.cat([prompt, next_token], dim=1)


    # Decode back to text 
    generated_text = "".join(chr(int(t)) for t in prompt[0].tolist())
    print(generated_text)

