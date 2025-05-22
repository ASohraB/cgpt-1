import torch.nn as nn

#import torch.nn as nn
from utils.helpers import cLinear, cMultiHeadAttention, cLayerNorm, cMLP

class NanoGPTBlock(nn.Module):
    def __init__(self, embed_size, nheads, mlp_hidden_size):
        super().__init__()
        self.ln1 = cLayerNorm(embed_size)
        self.mha = cMultiHeadAttention(
            E_q=embed_size, E_k=embed_size, E_v=embed_size,
            E_total=embed_size, nheads=nheads
        )
        self.ln2 = cLayerNorm(embed_size)
        self.mlp = cMLP(embed_size, mlp_hidden_size, embed_size, num_layers=2)

    def forward(self, x):
        # Self-attention block
        attn_out = self.mha(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        # MLP block
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)
        return x

class GPT(nn.Module):
    def __init__(self, input_size, embed_size, output_size, nheads, mlp_hidden_size, num_layers):
        super().__init__()
        self.embed = cLinear(input_size, embed_size)
        self.blocks = nn.Sequential(
            *[NanoGPTBlock(embed_size, nheads, mlp_hidden_size) for _ in range(num_layers)]
        )
        self.head = cLinear(embed_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size) or (batch, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add seq_len=1 if needed
        x = self.embed(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.squeeze(1)  # Remove seq_len if it was added