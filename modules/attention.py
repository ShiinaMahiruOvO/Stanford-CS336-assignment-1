import torch
import torch.nn as nn
from jaxtyping import Bool, Float
from torch import Tensor
from einops import einsum
from ..layers import softmax
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,
                query: Float[Tensor, "... q d_k"],
                key: Float[Tensor, "... k d_k"],
                value: Float[Tensor, "... k d_v"],
                mask: Bool[Tensor, "... q k"] = None
                ) -> Float[Tensor, "... q d_v"]:
        scale = math.sqrt(query.size(-1))
        attn_scores = einsum(query, key, "... q d_k, ... k d_k -> ... q k") / scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_probs = softmax(attn_scores, i=-1)
        output = einsum(attn_probs, value, "... q k, ... k d -> ... q d")
        return output
