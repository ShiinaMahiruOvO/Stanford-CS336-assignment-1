import torch
import torch.nn as nn
from ..layers import RMSNorm
from ..modules import CasualMultiheadSelfAttention, SwiGLUffn

class TransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 d_ff: int,
                 use_rope: bool,
                 max_seq_len: int = 2048,
                 theta: float = 10000.0,
                 device=None,
                 dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.ln1 = RMSNorm(d_model, **kwargs)
        self.attn = CasualMultiheadSelfAttention(d_model, 
                                                      num_heads,
                                                      use_rope=use_rope,
                                                      max_seq_len=max_seq_len,
                                                      theta=theta,
                                                      **kwargs)
        self.ln2 = RMSNorm(d_model, **kwargs)
        self.ffn = SwiGLUffn(d_model, d_ff, **kwargs)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        attn_out = self.attn(self.ln1(x), token_positions)
        x = x + attn_out
        ffn_out = self.ffn(self.ln2(x))
        return x + ffn_out
