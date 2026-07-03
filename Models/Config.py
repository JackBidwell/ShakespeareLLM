from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int
    embed_size: int = 384
    num_layers: int = 6
    num_heads: int = 6
    block_size: int = 384
    forward_expansion: int = 4
    dropout: float = 0.2
