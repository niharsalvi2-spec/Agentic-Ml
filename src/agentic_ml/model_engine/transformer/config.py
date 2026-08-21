"""
Transformer Model Configuration for Custom Agentic ML LLM.
"""

from dataclasses import dataclass

@dataclass
class GenixConfig:
    vocab_size: int = 50304      # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    max_seq_len: int = 1024     # Context length / maximum sequence length
    n_layer: int = 12           # Number of transformer layers
    n_head: int = 12            # Number of attention heads
    n_embd: int = 768           # Embedding dimensionality
    dropout: float = 0.0        # Dropout rate
    bias: bool = False          # Use bias in Linears and LayerNorms (False for faster/cleaner training)
