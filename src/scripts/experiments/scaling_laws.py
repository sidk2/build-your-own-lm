#!/usr/bin/env python
import argparse
import os
import sys
import time
from typing import Dict, List
import numpy as np
import pandas as pd
from datasets import load_dataset

from src.model.tokenizer import BPETokenizer

import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")


def fit_power_law(x, y):
    """Fits y = a * x^b by running linear regression on log-log data."""
    # Filter out zeros or negative values
    valid = (x > 0) & (y > 0)
    x_val = x[valid]
    y_val = y[valid]
    if len(x_val) < 2:
        return 0, 0
    coeffs = np.polyfit(np.log(x_val), np.log(y_val), 1)
    b = coeffs[0]  # power exponent
    a = np.exp(coeffs[1])  # multiplier
    return a, b


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scaling laws of the BPETokenizer."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments",
        help="Directory to save experimental results and plots",
    )
    parser.add_argument(
        "--vocab-sizes",
        type=str,
        default="200,400,800,1600, 3200, 12800",
        help="Comma-separated vocabulary sizes to test",
    )
    parser.add_argument(
        "--corpus-sizes",
        type=str,
        default="100, 500, 2500, 12500",
        help="Comma-separated corpus sizes (number of stories) to test",
    )
    args = parser.parse_args()

    # Parse grids
    vocab_sizes = [int(v) for v in args.vocab_sizes.split(",")]
    corpus_sizes = [int(c) for c in args.corpus_sizes.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("BPE Tokenizer Scaling Laws Experiment")
    print("=" * 60)
    print(f"Vocab size grid:  {vocab_sizes}")
    print(f"Corpus size grid: {corpus_sizes}")
    print(f"Saving plots to:  {args.output_dir}")
    print("-" * 60)

    # Load TinyStories
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    results = []

    # Run experiment grid
    for n in corpus_sizes:
        print(f"\nExtracting corpus of size {n} stories...")
        # Get raw stories
        subset = dataset.select(range(min(n, len(dataset))))
        stories = [item["text"] for item in subset]

        # Calculate corpus stats
        total_chars = sum(len(s) for s in stories)
        total_words = sum(len(s.split()) for s in stories)

        for v in vocab_sizes:
            print(f"  Training tokenizer with Vocab Size: {v} ... ", end="", flush=True)

            # Start timer
            start_time = time.perf_counter()
            tokenizer = BPETokenizer(vocab_size=v)
            tokenizer.train(stories)
            elapsed_time = time.perf_counter() - start_time

            # Calculate compression ratio
            # Tokenize the corpus
            total_tokens = 0
            for story in stories:
                total_tokens += len(tokenizer.encode(story))

            compression_ratio_words = total_words / max(1, total_tokens)
            compression_ratio_chars = total_chars / max(1, total_tokens)

            print(
                f"Done in {elapsed_time:.2f}s (Compression: {compression_ratio_words:.2f} words/tok)"
            )

            results.append(
                {
                    "corpus_size_stories": n,
                    "corpus_size_words": total_words,
                    "corpus_size_chars": total_chars,
                    "vocab_size": v,
                    "training_time_sec": elapsed_time,
                    "total_tokens": total_tokens,
                    "compression_words_per_token": compression_ratio_words,
                    "compression_chars_per_token": compression_ratio_chars,
                }
            )

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, "scaling_results.csv"), index=False)
    print(
        f"\nSaved raw results CSV to {os.path.join(args.output_dir, 'scaling_results.csv')}"
    )

    # --- Plotting ---
    print("\nGenerating scaling plots...")

    # Plot 1: Training Time vs. Vocab Size for various corpus sizes
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in corpus_sizes:
        sub = df[df["corpus_size_stories"] == n]
        ax.plot(
            sub["vocab_size"], sub["training_time_sec"], "o-", label=f"N={n} stories"
        )
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time vs. Vocabulary Size")
    ax.legend(title="Corpus Size")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "time_vs_vocab.png"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, "time_vs_vocab.pdf"))
    plt.close()

    # Plot 2: Training Time vs. Corpus Size for various vocab sizes
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for v in vocab_sizes:
        sub = df[df["vocab_size"] == v]
        ax.plot(
            sub["corpus_size_stories"], sub["training_time_sec"], "o-", label=f"V={v}"
        )
    ax.set_xlabel("Corpus Size (stories)")
    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time vs. Corpus Size")
    ax.legend(title="Vocab Size")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "time_vs_corpus.png"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, "time_vs_corpus.pdf"))
    plt.close()

    # Plot 3: Compression Ratio vs. Vocab Size
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for n in corpus_sizes:
        sub = df[df["corpus_size_stories"] == n]
        ax.plot(
            sub["vocab_size"],
            sub["compression_words_per_token"],
            "s-",
            label=f"N={n} stories",
        )
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Compression Ratio (words/token)")
    ax.set_title("Compression Ratio vs. Vocabulary Size")
    ax.legend(title="Corpus Size")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "compression_vs_vocab.png"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, "compression_vs_vocab.pdf"))
    plt.close()

    # Plot 4: 2D Heatmap of Training Time
    pivot_df = df.pivot(
        index="corpus_size_stories", columns="vocab_size", values="training_time_sec"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    cax = ax.imshow(pivot_df.values, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(vocab_sizes)))
    ax.set_xticklabels(vocab_sizes)
    ax.set_yticks(np.arange(len(corpus_sizes)))
    ax.set_yticklabels(corpus_sizes)
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Corpus Size (stories)")
    ax.set_title("Training Time Heatmap (seconds)")
    fig.colorbar(cax, label="Seconds")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "time_heatmap.png"), dpi=300)
    plt.savefig(os.path.join(args.output_dir, "time_heatmap.pdf"))
    plt.close()

    # --- Fitting and Scaling Analysis ---
    print("\n" + "=" * 60)
    print("ANALYSIS OF SCALING LAWS")
    print("=" * 60)

    # 1. Training Time vs. Corpus Size: T ~ c * N^a
    print("Fitting Training Time vs. Corpus Size (T ~ c * N^a):")
    for v in vocab_sizes:
        sub = df[df["vocab_size"] == v]
        c, a = fit_power_law(
            sub["corpus_size_stories"].values, sub["training_time_sec"].values
        )
        print(
            f"  For Vocab Size V={v:4d}: Exponent a = {a:.3f}  (Multiplier c = {c:.2e})"
        )

    # 2. Training Time vs. Vocab Size: T ~ c * V^b
    print("\nFitting Training Time vs. Vocab Size (T ~ c * V^b):")
    for n in corpus_sizes:
        sub = df[df["corpus_size_stories"] == n]
        c, b = fit_power_law(sub["vocab_size"].values, sub["training_time_sec"].values)
        print(
            f"  For Corpus Size N={n:4d}: Exponent b = {b:.3f}  (Multiplier c = {c:.2e})"
        )

    # 3. Compression Ratio vs. Vocab Size: C ~ c * V^d
    print("\nFitting Compression Ratio (words/token) vs. Vocab Size (C ~ c * V^d):")
    for n in corpus_sizes:
        sub = df[df["corpus_size_stories"] == n]
        c, d = fit_power_law(
            sub["vocab_size"].values, sub["compression_words_per_token"].values
        )
        print(
            f"  For Corpus Size N={n:4d}: Exponent d = {d:.3f}  (Multiplier c = {c:.2e})"
        )

    print(
        "\nScaling analysis complete! All plots saved as PNG and PDF in:",
        args.output_dir,
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
