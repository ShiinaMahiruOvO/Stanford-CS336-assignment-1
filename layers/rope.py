import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        token_positions = torch.arange(max_seq_len, device=device).float()
        freqs = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        angles = token_positions[:, None] * freqs[None, :]
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
        if device is not None:
            self.to(device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions].unsqueeze(1)
        sin = self.sin_cached[token_positions].unsqueeze(1)
        
        x_even, x_odd = x[..., ::2], x[..., 1::2]
        x_complex = torch.complex(x_even, x_odd)
        rot = torch.complex(cos, sin)
        x_complex = x_complex * rot
        
        return torch.view_as_real(x_complex).flatten(-2)
