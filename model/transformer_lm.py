import torch
import torch.nn as nn
from ..layers import Embedding, RMSNorm, Linear
from .transformer_block import TransformerBlock

class TransformerLM(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 context_length: int,
                 num_layers: int,
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 use_rope: bool,
                 theta: float = 10000.0,
                 device=None,
                 dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.token_embeddings = Embedding(vocab_size, d_model, **kwargs)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model,
                             num_heads,
                             d_ff,
                             use_rope=use_rope,
                             max_seq_len=context_length,
                             theta=theta,
                             **kwargs)
            for _ in range(num_layers)
        ])
        self.context_length = context_length
        self.ln_final = RMSNorm(d_model, **kwargs)
        self.lm_head = Linear(d_model, vocab_size, **kwargs)
        self.register_buffer("pos_cache", torch.arange(context_length).unsqueeze(0), persistent=False)
        
    def forward(self, token_ids: torch.Tensor):
        b, s = token_ids.shape
        if s > self.context_length:
            raise ValueError(f"seq_len {s} exceeds context_length {self.context_length}")
        x = self.token_embeddings(token_ids)
        pos = self.pos_cache[:, :s].expand(b, s)
        for block in self.layers:
            x = block(x, pos)
        x = self.ln_final(x)
        out = self.lm_head(x)
        return out