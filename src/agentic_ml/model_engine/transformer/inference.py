"""
Inference generation module for Custom Transformer Foundation Model.
"""

import os
import torch
from pathlib import Path
from src.agentic_ml.model_engine.transformer.model import GenixTransformerModel
from src.agentic_ml.model_engine.transformer.config import GenixConfig
from src.agentic_ml.model_engine.transformer.tokenizer import GPT2Tokenizer

class TransformerInferenceEngine:
    def __init__(self, checkpoint_path: str = "artifacts/models/genix_baseline.pt", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = GPT2Tokenizer()
        self.config = GenixConfig()
        self.model = GenixTransformerModel(self.config).to(self.device)
        
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.8, top_k: int = 40) -> str:
        tokens = self.tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            x_cond = x if x.size(1) <= self.config.max_seq_len else x[:, -self.config.max_seq_len:]
            logits, _ = self.model(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)
            
            if idx_next.item() == self.tokenizer.eot_token:
                break
                
        out_tokens = x[0].tolist()
        return self.tokenizer.decode(out_tokens)
