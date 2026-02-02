import torch
import numpy as np
from .training_utils import *
from dataclasses import dataclass
from .experiment_logger import ExperimentLogger


@dataclass
class TrainConfig:
    # ========== data parameters ==========
    train_data: str
    valid_data: str
    batch_size: int = 16
    context_length: int = 128
    device: str = "mps"
    dtype: torch.dtype = torch.float32
    
    # ========== model parameters ==========
    vocab_size: int = 10000
    num_layers: int = 2
    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 512
    use_rope: bool = True
    theta: float = 10000.0
    
    # ========== optimizer parameters ==========
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    
    # ========== gradient clipping ==========
    max_l2_norm: float = 1.0
    
    # ========== lr schedule ==========
    max_lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_iters: int = 200
    cos_cycle_iters: int = 5000
    
    # ========== training control ==========
    max_iters: int = 2000
    eval_interval: int = 10
    save_interval: int = 1000
    checkpoint_path: str = "checkpoint.pt"
    
    # ========== logging ==========
    log_path: str | None = None
    log_print_every_sec: float | None = None


def train(config: TrainConfig,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          start_it: int = 0,
          logger: ExperimentLogger | None = None):
    train_data = np.memmap(config.train_data, dtype=np.uint16, mode="r")
    valid_data = np.memmap(config.valid_data, dtype=np.uint16, mode="r")

    if logger is None and config.log_path is not None:
        logger = ExperimentLogger(config.log_path, print_every=config.log_print_every_sec)
    
    try:
        for it in range(start_it, config.max_iters):
            X, Y = get_batch(train_data,
                             config.batch_size,
                             config.context_length,
                             config.device)
            logits = model(X)
            loss = cross_entropy_loss(logits, Y)
            
            lr = lr_schedule(
                it,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                warmup_iters=config.warmup_iters,
                cos_cycle_iters=config.cos_cycle_iters,
            )
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), config.max_l2_norm)
            optimizer.step()
            
            if it % config.eval_interval == 0:
                Xv, Yv = get_batch(valid_data, config.batch_size, config.context_length, config.device)
                with torch.no_grad():
                    val_logits = model(Xv)
                    val_loss = cross_entropy_loss(val_logits, Yv)
                print(f"[iter {it}] train loss={loss.item():.4f}, val loss={val_loss.item():.4f}")
                
                if logger is not None:
                    logger.log(it, train_loss=loss.item(), val_loss=val_loss.item())

            if it % config.save_interval == 0:
                save_checkpoint(model, optimizer, it, config.checkpoint_path)
    finally:
        if logger is not None:
            logger.save()


def resume_train(config: TrainConfig,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 src: str,
                 logger: ExperimentLogger | None = None):
    iteration = load_checkpoint(model, optimizer, src)
    
    train(config, model, optimizer, start_it=iteration + 1, logger=logger)
