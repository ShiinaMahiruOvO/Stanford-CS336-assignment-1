import torch
import torch.nn as nn
from ..layers import Linear, RotaryPositionalEmbedding
from .attention import ScaledDotProductAttention
from einops import rearrange

class CasualMultiheadSelfAttention(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 use_rope: bool = True,
                 max_seq_len: int = 2048,
                 theta: float = 10000.0,
                 device=None,
                 dtype=None):
        assert d_model % num_heads == 0
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        kwargs = {'device': device, 'dtype': dtype}
        self.q_proj, self.k_proj, self.v_proj, self.output_proj = [Linear(d_model, d_model, **kwargs) for _ in range(4)]
        self.attention = ScaledDotProductAttention()
        # lower triangular mask keeps current and past positions while masking future tokens
        self.mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))
        self.mask = self.mask[None, None, :, :]
        
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
            
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        q, k, v = [rearrange(proj(x), "b l (h d) -> b h l d", h=self.num_heads)
                   for proj in [self.q_proj, self.k_proj, self.v_proj]]
        if self.use_rope:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        _, L, _ = x.shape    # B: batch, L: seq_len, D: embedding_dim
        out = self.attention(q, k, v, self.mask[:, :, :L, :L].to(x.device))
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.output_proj(out)
