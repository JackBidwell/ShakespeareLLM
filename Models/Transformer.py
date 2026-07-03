import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()

        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.dropout = dropout

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_len, embed_size = x.shape

        values = self.values(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.keys(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.queries(x).view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            queries, keys, values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).reshape(N, seq_len, embed_size)

        return self.resid_dropout(self.fc_out(out))


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout):
        super().__init__()

        self.attention = SelfAttention(embed_size, num_heads, dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.block_size = config.block_size

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding = nn.Embedding(config.block_size, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config.embed_size, config.num_heads, config.forward_expansion, config.dropout)
            for _ in range(config.num_layers)
        ])

        self.norm_out = nn.LayerNorm(config.embed_size)
        self.fc_out = nn.Linear(config.embed_size, config.vocab_size, bias=False)

        # weight tying: input embedding and output projection share weights
        self.fc_out.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # scaled init for residual projections, GPT-2 style
        for name, p in self.named_parameters():
            if name.endswith("fc_out.weight") and p is self.token_embedding.weight:
                continue
            if name.endswith("attention.fc_out.weight") or name.endswith("feedforward.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        N, seq_len = x.shape

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(N, seq_len)

        x = self.dropout(self.token_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)
        logits = self.fc_out(x)

        return logits
