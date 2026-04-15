import torch
import torch.nn as nn
import torch.nn.functional as F        

class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()

        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_len, embed_size = x.shape

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # split into heads
        values = values.view(N, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(N, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(N, seq_len, self.num_heads, self.head_dim)

        # attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # causal mask (no looking ahead)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
        energy = energy.masked_fill(mask == 0, float("-inf"))

        attention = F.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        out = torch.einsum("nhqk,nkhd->nqhd", [attention, values])

        out = out.reshape(N, seq_len, self.embed_size)

        return self.fc_out(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion):
        super().__init__()

        self.attention = SelfAttention(embed_size, num_heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)

        forward = self.feedforward(x)
        out = self.norm2(forward + x)

        return out
    
class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size=128,
        num_layers=4,
        num_heads=4,
        block_size=256,
        forward_expansion=4
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(block_size, embed_size)

        self.layers = nn.Sequential(
            *[
                TransformerBlock(embed_size, num_heads, forward_expansion)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        N, seq_len = x.shape

        positions = torch.arange(0, seq_len).expand(N, seq_len).to(x.device)

        x = self.token_embedding(x) + self.position_embedding(positions)

        x = self.layers(x)

        logits = self.fc_out(x)

        return logits