import argparse
import time
from pathlib import Path
from ..model import TransformerLM
from ..training import TrainConfig, train, resume_train, AdamW, ExperimentLogger


def parse_args_to_config() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Training configuration")

    # Required
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)

    # Optional
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--log_print_every_sec", type=float, default=30.0)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main():
    config = parse_args_to_config()
    if config.log_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = Path("logs") / f"train_{timestamp}.json"
    else:
        log_path = Path(config.log_path)

    logger = ExperimentLogger(log_path, print_every=config.log_print_every_sec)
    # Persist the resolved path back to the config so resumed runs reuse the same file.
    config.log_path = str(log_path)

    model = TransformerLM(
        config.vocab_size,
        config.context_length,
        config.num_layers,
        config.d_model,
        config.num_heads,
        config.d_ff,
        config.use_rope,
        config.theta,
        config.device,
        config.dtype
    )
    optimizer = AdamW(
        params=model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay
    )
    
    ckpt = Path(config.checkpoint_path)
    if ckpt.is_file():
        resume_train(config, model, optimizer, ckpt, logger=logger)
    else:
        train(config, model, optimizer, logger=logger)
        
        
if __name__ == "__main__":
    main()
