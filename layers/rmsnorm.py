import torch
import torch.nn as nn
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        nn.init.ones_(self.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): shape (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: the same shape as input
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).rsqrt()
        gain = rearrange(self.weight, "d_model -> 1 1 d_model")
        result = x * rms * gain
        return result.to(in_dtype)
        
        