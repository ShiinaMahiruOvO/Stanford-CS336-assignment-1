import torch

def softmax(x: torch.Tensor, i: int = -1):
    """Apply the softmax function to the i_th dimension of x"""
    x_max = x.max(dim=i, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=i, keepdim=True)
    return x_exp / x_sum