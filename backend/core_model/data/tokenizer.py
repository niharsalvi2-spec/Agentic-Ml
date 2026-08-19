import tiktoken

class GenixTokenizer:
    """
    A unified tokenizer interface for Genix.
    Wraps tiktoken for production-grade BPE speed while allowing custom token injection.
    """
    def __init__(self, encoding_name="gpt2"):
        # We use gpt2 encoding by default as it matches our 50257 vocab size config
        self.enc = tiktoken.get_encoding(encoding_name)
        
        # Example of how we could inject custom tokens if training from scratch
        # self.enc = tiktoken.Encoding(
        #     name="genix_custom",
        #     pat_str=self.enc._pat_str,
        #     mergeable_ranks=self.enc._mergeable_ranks,
        #     special_tokens={**self.enc._special_tokens, "<think>": 50257, "</think>": 50258}
        # )
        
    def encode(self, text: str, allowed_special="all") -> list[int]:
        """Convert a string into a list of integer tokens."""
        return self.enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: list[int]) -> str:
        """Convert a list of integer tokens back into a string."""
        return self.enc.decode(tokens)

    @property
    def vocab_size(self):
        return self.enc.n_vocab
