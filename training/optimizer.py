import torch
import math
from collections.abc import Callable
from typing import Optional

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = (
                group['lr'], group['betas'], group['eps'], group['weight_decay']
            )
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                state['step'] += 1
                t = state['step']
                m, v = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas[0], betas[1]
                
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                adjusted_lr = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data.add_(-m / (v.sqrt() + eps), alpha=adjusted_lr)
                if weight_decay != 0:
                    p.data.add_(-p.data, alpha=weight_decay*lr)
                    
        return loss