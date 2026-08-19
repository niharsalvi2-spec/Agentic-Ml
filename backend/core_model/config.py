from dataclasses import dataclass

@dataclass
class GenixConfig:
    """
    Configuration class for the Genix LLM Architecture.
    Designed to be highly scalable from 'Nano' local testing to 'Ultra' cluster deployment.
    """
    vocab_size: int = 50257     # Default GPT-2/Tiktoken vocab size
    max_seq_len: int = 2048     # Context Window Size (Attention Mechanism scope)
    n_embd: int = 768           # Dimensionality of the embeddings and hidden states
    n_head: int = 12            # Number of attention heads
    n_layer: int = 12           # Number of transformer blocks
    dropout: float = 0.1        # Dropout probability
    bias: bool = False          # True: bias in Linears and LayerNorms, False: a bit better and faster
    
    @classmethod
    def get_nano_config(cls):
        """A lightweight configuration for local testing and CPU execution."""
        return cls(
            n_layer=4,
            n_head=4,
            n_embd=256,
            max_seq_len=512
        )
