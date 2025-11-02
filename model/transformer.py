import torch
from model.transformer_block import TransformerBlock

class MiniTransformer(torch.nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, heads=4, d_ff=256):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList(
            [TransformerBlock(d_model, heads, d_ff) for _ in range(num_layers)]
        )
        self.fc_out = torch.nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.fc_out(x)
        return logits
