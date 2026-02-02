import torch
import numpy as np
import numpy.typing as npt
import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO
from einops import rearrange

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    """ Args:
        logits (torch.Tensor): (batch_size, context_length, vocab_size)
        targets (torch.Tensor): (batch_size, context_length)
    """
    logits = rearrange(logits, "B L V -> (B L) V")
    targets = rearrange(targets, "B L -> (B L)")
    
    log_sum_exp = torch.logsumexp(logits, dim=-1)  
    logit_targets = logits[torch.arange(len(targets)), targets]
    loss = log_sum_exp - logit_targets
    return loss.mean()


def lr_schedule(it: int,
                max_lr: float, min_lr: float, 
                warmup_iters: int, cos_cycle_iters: int) -> float:
    if it < 0:
        raise ValueError(f"Invalid iteration number: {it}")
    if it < warmup_iters:
        return it * max_lr / warmup_iters
    elif it <= cos_cycle_iters:
        progress = (it - warmup_iters) / (cos_cycle_iters - warmup_iters)
        cosine = 0.5 * (1 + math.cos(progress * math.pi))
        return min_lr + cosine * (max_lr - min_lr)
    elif it > cos_cycle_iters:
        return min_lr
    

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    eps = 1e-6
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
            
            
def get_batch(
    x: npt.NDArray, batch_size: int, context_length: int, device: str = 'mps'
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(x) - context_length, size=batch_size)
    X = np.stack([x[i : i + context_length] for i in starts])
    Y = np.stack([x[i + 1: i + 1 + context_length] for i in starts])
    
    X = torch.from_numpy(X).long().to(device)
    Y = torch.from_numpy(Y).long().to(device)
    return X, Y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes] = "checkpoint.pt",
):
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'iteration': iteration
    }
    torch.save(checkpoint, out)
    
    
def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    src: str | os.PathLike | BinaryIO | IO[bytes] = "checkpoint.pt",
) -> int:
    checkpoint = torch.load(src)
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    iteration = checkpoint['iteration']
    
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    return iteration