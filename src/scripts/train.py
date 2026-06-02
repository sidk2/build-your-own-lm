#!/usr/bin/env python
import argparse
import math
import os
import sys
import time
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from model.gpt import GPT
from model.tokenizer import BPETokenizer
from data.dataset import TinyStoriesDataset


def get_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    # Weight decay only for parameters >= 2D (weights of Linear/Embedding layers)
    # 1D parameters like biases and layernorm weights do not get weight decayed

    # This was recommended to me by Gemini, but I don't fully understand why its a good idea yet.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"Weight Decay configuration:")
    print(
        f"  Decayed tensors: {len(decay_params):,} tensors | {num_decay_params:,} parameters"
    )
    print(
        f"  Non-decayed tensors: {len(nodecay_params):,} tensors | {num_nodecay_params:,} parameters"
    )

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


def get_lr(
    it: int, warmup_iters: int, lr_decay_iters: int, max_lr: float, min_lr: float
) -> float:
    # Implementing Cosine learning rate decay with linear warmup.
    # Learning rate increases linearly for `warmup_iters` steps
    # and then decreases following a cosine curve until `lr_decay_iters` steps
    # after which it stays at `min_lr`.

    if it < warmup_iters:
        return max_lr * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_batch_generator(loader: DataLoader, device: str):
    while True:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module, eval_iters: int, train_iter, val_iter
) -> Dict[str, float]:
    out = {}
    model.eval()
    for split, iterator in [("train", train_iter), ("val", val_iter)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = next(iterator)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def sample_text(
    model: torch.nn.Module,
    tokenizer: BPETokenizer,
    device: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [tokenizer.rev_vocab.get("<unk>", 0)]
    x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)
    eos_id = tokenizer.rev_vocab.get("<eos>", None)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50, eos_token_id=eos_id)
    return tokenizer.decode(y[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Train Mini-GPT on TinyStories")
    # Model config
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Dimensionality of embeddings/hidden state",
    )
    parser.add_argument(
        "--num-heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of Transformer blocks"
    )
    parser.add_argument(
        "--use-bias",
        action="store_true",
        default=False,
        help="Enable bias in attention and MLP layers",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Maximum sequence/context length",
    )

    # Training config
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max-iters", type=int, default=5000, help="Total training iterations/steps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=6e-4, help="Max learning rate"
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=6e-5,
        help="Min learning rate after cosine decay",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=500,
        help="Learning rate linear warmup steps",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay parameter"
    )
    parser.add_argument(
        "--grad-clip", type=float, default=1.0, help="Clip gradients at this value"
    )

    # Evaluation config
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=500,
        help="Step interval for validation evaluation",
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=100,
        help="Number of batches to evaluate loss over",
    )
    parser.add_argument(
        "--sample-prompt",
        type=str,
        default="Once upon a time, there was a little bird",
        help="Prompt to generate samples from",
    )

    # System config
    parser.add_argument(
        "--device", type=str, default="auto", help="auto, cuda, mps, or cpu"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="data/tokenizer.json",
        help="Path to BPE tokenizer json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/out",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=50000,
        help="Limit number of TinyStories stories (useful for tests)",
    )
    parser.add_argument(
        "--no-pre-tokenize",
        action="store_true",
        default=False,
        help="Do not pre-tokenize dataset (slower, but saves RAM)",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    args = parser.parse_args()

    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("Mini-GPT Training Pipeline")
    print("=" * 60)
    print(f"Device:                 {device}")
    print(
        f"Model Configuration:    d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}, use_bias={args.use_bias}"
    )
    print(
        f"Training parameters:    lr={args.learning_rate}, min_lr={args.min_lr}, warmup_iters={args.warmup_iters}"
    )
    print(f"Batch Size:             {args.batch_size} | Max Steps: {args.max_iters}")
    print(f"Output Directory:       {args.out_dir}")
    print("-" * 60)

    # Seed initialization
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load Tokenizer
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer json file not found at '{args.tokenizer_path}'.")
        print("Please train a tokenizer first by running run_tokenizer.py.")
        sys.exit(1)

    print(f"Loading tokenizer from '{args.tokenizer_path}'...")
    tokenizer = BPETokenizer(
        vocab_size=50000
    )
    tokenizer.load(args.tokenizer_path)
    vocab_size = len(tokenizer.vocabulary)
    print(f"Tokenizer loaded successfully. Vocabulary size: {vocab_size}")

    # Load Dataset and DataLoader
    print("Loading datasets...")
    pre_tokenize = not args.no_pre_tokenize

    train_dataset = TinyStoriesDataset(
        split="train",
        tokenizer=tokenizer,
        block_size=args.context_length,
        limit_samples=args.limit_samples,
        pre_tokenize=pre_tokenize,
    )

    val_limit = args.limit_samples // 10 if args.limit_samples is not None else None
    val_dataset = TinyStoriesDataset(
        split="validation",
        tokenizer=tokenizer,
        block_size=args.context_length,
        limit_samples=val_limit,
        pre_tokenize=pre_tokenize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if "cuda" in device else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if "cuda" in device else False,
    )

    train_iter = get_batch_generator(train_loader, device)
    val_iter = get_batch_generator(val_loader, device)

    print("Initializing GPT model...")
    model = GPT(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_bias=args.use_bias,
        masked=True,
        context_length=args.context_length,
    )
    model.to(device)

    optimizer = get_optimizer(
        model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
    )

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_loss = float("inf")

    print("Starting training...")
    t0 = time.time()

    for it in range(args.max_iters):
        lr = get_lr(
            it=it,
            warmup_iters=args.warmup_iters,
            lr_decay_iters=args.max_iters,
            max_lr=args.learning_rate,
            min_lr=args.min_lr,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, args.eval_iters, train_iter, val_iter)
            print(
                f"Step {it:5d}: Train loss {losses['train']:.4f} | Val loss {losses['val']:.4f} | LR {lr:.2e}"
            )

            print(f"--- Generation sample at step {it} ---")
            sample_out = sample_text(
                model, tokenizer, device, args.sample_prompt, max_new_tokens=64
            )
            print(sample_out)
            print("-" * 60)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab_size": vocab_size,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "use_bias": args.use_bias,
                "context_length": args.context_length,
                "iter": it,
                "val_loss": losses["val"],
                "args": args.__dict__,
            }

            latest_ckpt_path = os.path.join(args.out_dir, "ckpt_latest.pt")
            torch.save(checkpoint, latest_ckpt_path)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                best_ckpt_path = os.path.join(args.out_dir, "ckpt_best.pt")
                torch.save(checkpoint, best_ckpt_path)
                print(
                    f"  New best validation loss! Saved checkpoint to {best_ckpt_path}"
                )

        x, y = next(train_iter)

        t_step_start = time.time()
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for improved stability.
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        t_step_end = time.time()

        if it % 10 == 0:
            lossf = loss.item()
            dt = t_step_end - t_step_start
            print(
                f"Step {it:5d} | Loss {lossf:.4f} | Time per step: {dt*1000:.1f}ms | LR {lr:.2e}"
            )

    total_time = time.time() - t0
    print(f"Training complete! Total training time: {total_time/60:.2f} minutes.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
