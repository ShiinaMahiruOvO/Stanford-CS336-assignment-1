import argparse
import json
import time
from array import array
from pathlib import Path

import torch

from ..model import TransformerLM
from ..tokenizer import BPETokenizer, _UNICODE_TO_BYTES
from ..training import AdamW, TrainConfig, resume_train, train


def _token_bytes(token: str) -> bytes:
    """Convert a GPT-2 style unicode token to raw bytes using the lookup table."""
    out = bytearray()
    for ch in token:
        mapped = _UNICODE_TO_BYTES.get(ch)
        if mapped is not None:
            out.append(mapped)
        else:
            # Fallback: use UTF-8 bytes for characters outside the GPT-2 alphabet.
            out.extend(ch.encode("utf-8"))
    return bytes(out)


def load_tokenizer(
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    """Instantiate a BPETokenizer from TinyStories vocab/merges dumps."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)

    # Files ship as {str(idx): token}; convert to {idx: bytes}.
    vocab_items = sorted(((int(idx), tok) for idx, tok in vocab_json.items()), key=lambda p: p[0])
    vocab_bytes: dict[int, bytes] = {idx: _token_bytes(tok) for idx, tok in vocab_items}
    existing_bytes = set(vocab_bytes.values())

    merges: list[tuple[bytes, bytes]] = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip("\n")
            if not cleaned or cleaned.startswith("#"):
                continue
            parts = cleaned.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"Unable to parse merge line: {cleaned!r}")
            a, b = parts
            merges.append((_token_bytes(a), _token_bytes(b)))

    # Ensure all single-byte tokens are present (needed for streaming encode).
    next_idx = max(vocab_bytes.keys(), default=-1) + 1
    for i in range(256):
        b = bytes([i])
        if b not in existing_bytes:
            vocab_bytes[next_idx] = b
            existing_bytes.add(b)
            next_idx += 1

    if special_tokens:
        for tok in special_tokens:
            b = tok.encode("utf-8")
            if b not in existing_bytes:
                vocab_bytes[next_idx] = b
                existing_bytes.add(b)
                next_idx += 1

    return BPETokenizer(special_tokens=special_tokens, vocab=vocab_bytes, merges=merges)


def tokenize_corpus(
    tokenizer: BPETokenizer,
    source_path: Path,
    output_path: Path,
    chunk_size: int = 1_000_000,
) -> None:
    """Stream-encode a text corpus into uint16 token ids."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buf = array("H")
    with open(source_path, "r", encoding="utf-8") as fin, open(output_path, "wb") as fout:
        for token_id in tokenizer.encode_iterable(fin):
            if token_id >= 2**16:
                raise ValueError(
                    f"Token id {token_id} exceeds uint16 range. "
                    "Consider increasing the dtype width."
                )
            buf.append(token_id)
            if len(buf) >= chunk_size:
                buf.tofile(fout)
                buf = array("H")
        if buf:
            buf.tofile(fout)


def ensure_bin_file(
    tokenizer: BPETokenizer,
    text_path: Path,
    bin_path: Path,
    force: bool = False,
) -> None:
    """Create the binary token file if missing or if force==True."""
    if bin_path.exists() and not force:
        return
    print(f"[tokenize] {text_path} -> {bin_path}")
    tokenize_corpus(tokenizer, text_path, bin_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer LM on TinyStories.")

    parser.add_argument("--train_text", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--valid_text", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--train_bin", type=Path, default=None)
    parser.add_argument("--valid_bin", type=Path, default=None)
    parser.add_argument("--tokenizer_vocab", type=Path, default=Path("tinystories_10k_vocab.json"))
    parser.add_argument("--tokenizer_merges", type=Path, default=Path("tinystories_10k_merges.txt"))
    parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="Optional list of special tokens to reserve.",
    )
    parser.add_argument("--force_tokenize", action="store_true", help="Rebuild .bin files even if present.")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)

    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--use_rope", dest="use_rope", action="store_true")
    parser.add_argument("--no_rope", dest="use_rope", action="store_false")
    parser.set_defaults(use_rope=True)
    parser.add_argument("--theta", type=float, default=10000.0)

    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/tinystories.pt"))
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint if available.")

    parser.add_argument("--log_path", type=Path, default=None)
    parser.add_argument("--log_print_every_sec", type=float, default=30.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = load_tokenizer(
        vocab_path=args.tokenizer_vocab,
        merges_path=args.tokenizer_merges,
        special_tokens=args.special_tokens,
    )
    vocab_size = len(tokenizer.vocab)
    print(f"[setup] Loaded tokenizer with vocab size {vocab_size}.")

    train_bin = args.train_bin or args.train_text.with_suffix(".bin")
    valid_bin = args.valid_bin or args.valid_text.with_suffix(".bin")

    ensure_bin_file(tokenizer, args.train_text, train_bin, force=args.force_tokenize)
    ensure_bin_file(tokenizer, args.valid_text, valid_bin, force=args.force_tokenize)

    checkpoint_path = args.checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    if args.log_path is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = Path("logs") / f"tinystories_{timestamp}.json"
    else:
        log_path = args.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[logging] Writing log to {log_path}.")

    dtype = getattr(torch, args.dtype)
    config = TrainConfig(
        train_data=str(train_bin),
        valid_data=str(valid_bin),
        batch_size=args.batch_size,
        context_length=args.context_length,
        device=args.device,
        dtype=dtype,
        vocab_size=vocab_size,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        use_rope=args.use_rope,
        theta=args.theta,
        lr=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        checkpoint_path=str(checkpoint_path),
    )
    config.log_path = str(log_path)
    config.log_print_every_sec = args.log_print_every_sec

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
        config.dtype,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    if args.resume and checkpoint_path.is_file():
        print(f"[train] Resuming from checkpoint at {checkpoint_path}.")
        resume_train(config, model, optimizer, str(checkpoint_path))
    else:
        print("[train] Starting fresh training run.")
        train(config, model, optimizer)


if __name__ == "__main__":
    main()
