import torch
import torch.nn as nn
from collections import OrderedDict
from ..layers import Linear


def round_to_multiple_of_64(x: int) -> int:
    """Round up to nearest multiple of 64."""
    return int((x + 63) // 64 * 64)


def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))


class SwiGLUffn(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model= d_model
        self.d_ff = d_ff if d_ff % 64 ==0 else round_to_multiple_of_64(d_model * 8 // 3)
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(SiLU(self.w1(x)) * self.w3(x))

