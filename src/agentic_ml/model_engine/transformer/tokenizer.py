"""
Tokenizer abstraction for Custom Transformer Model.
"""

import tiktoken

class GPT2Tokenizer:
    """
    Standard GPT-2 / tiktoken BPE tokenizer wrapper.
    """
    def __init__(self):
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self.enc._special_tokens['<|endoftext|>']

    def encode(self, text: str) -> list[int]:
        return self.enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, tokens: list[int]) -> str:
        return self.enc.decode(tokens)
