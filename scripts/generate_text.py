import argparse
from pathlib import Path
from typing import List

import torch

from ..model import TransformerLM
from ..training import load_checkpoint
from ..training.decode import decode
from ..training.optimizer import AdamW  # type: ignore  # Needed only for checkpoint loading
from .train_tinystories import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained Transformer LM.")

    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/tinystories.pt"))
    parser.add_argument("--tokenizer_vocab", type=Path, default=Path("tinystories_10k_vocab.json"))
    parser.add_argument("--tokenizer_merges", type=Path, default=Path("tinystories_10k_merges.txt"))
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])

    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=1)

    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--use_rope", dest="use_rope", action="store_true")
    parser.add_argument("--no_rope", dest="use_rope", action="store_false")
    parser.set_defaults(use_rope=True)
    parser.add_argument("--theta", type=float, default=10000.0)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float32", "float16", "bfloat16"),
    )

    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def build_model(
    vocab_size: int,
    args: argparse.Namespace,
) -> TransformerLM:
    dtype = resolve_dtype(args.dtype)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=args.use_rope,
        theta=args.theta,
        device=args.device,
        dtype=dtype,
    )
    return model


def load_model_weights(model: TransformerLM, checkpoint_path: Path, device: str) -> int:
    optimizer = AdamW(model.parameters())  # Dummy optimizer so `load_checkpoint` can restore state.
    iteration = load_checkpoint(model, optimizer, str(checkpoint_path))
    model.to(device)
    model.eval()
    return iteration


def ensure_prompt_tokens(tokenizer, prompt: str, special_tokens: List[str]) -> List[int]:
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        for tok in special_tokens:
            encoded = tokenizer.encode(tok)
            if encoded:
                token_ids = encoded
                break
    return token_ids


def main() -> None:
    args = parse_args()

    tokenizer = load_tokenizer(
        vocab_path=args.tokenizer_vocab,
        merges_path=args.tokenizer_merges,
        special_tokens=args.special_tokens,
    )
    model = build_model(vocab_size=len(tokenizer.vocab), args=args)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    iteration = load_model_weights(model, args.checkpoint, args.device)
    print(f"[info] Loaded checkpoint from {args.checkpoint} (iteration {iteration}).")

    prompt_tokens = ensure_prompt_tokens(tokenizer, args.prompt, args.special_tokens)
    if len(prompt_tokens) >= args.context_length:
        raise ValueError(
            f"Prompt is too long ({len(prompt_tokens)} tokens) for context length {args.context_length}."
        )

    for sample_idx in range(args.num_samples):
        generated = decode(
            model=model,
            tokenizer=tokenizer,
            prompt=list(prompt_tokens),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )
        print("=" * 80)
        print(f"[sample {sample_idx + 1}]")
        print(generated)


if __name__ == "__main__":
    main()
