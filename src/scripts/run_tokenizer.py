#!/usr/bin/env python
import argparse
import os
import sys
from typing import List

# Ensure project root is in path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.model.tokenizer import BPETokenizer
from src.data.dataset import TinyStoriesDataset
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Train and test BPETokenizer on Hugging Face TinyStories dataset."
    )
    parser.add_argument(
        "--vocab-size", type=int, default=50000, help="Vocabulary size for BPE"
    )
    parser.add_argument(
        "--num-train-stories",
        type=int,
        default=50000,
        help="Number of stories to train the BPE tokenizer on",
    )
    parser.add_argument(
        "--num-verify-stories",
        type=int,
        default=50,
        help="Number of stories to verify tokenization on",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="data/tokenizer.json",
        help="Path to save the trained tokenizer",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Context/block size for the dataset loader",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TinyStories Tokenizer Runner")
    print("=" * 60)
    print(f"BPE Vocab Size: {args.vocab_size}")
    print(f"Training Stories count: {args.num_train_stories}")
    print(f"Tokenizer Output Path: {args.tokenizer_path}")
    print("-" * 60)

    print("Loading training stories...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

    train_subset = raw_dataset.select(
        range(min(args.num_train_stories, len(raw_dataset)))
    )
    corpus = [item["text"] for item in train_subset]

    print("Training BPETokenizer...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus)
    print(f"Training complete. Final vocabulary size: {len(tokenizer.vocabulary)}")

    print(f"Saving tokenizer to '{args.tokenizer_path}'...")
    tokenizer.save(args.tokenizer_path)

    print("Initializing TinyStoriesDataset...")
    dataset = TinyStoriesDataset(
        split="train",
        tokenizer=tokenizer,
        block_size=args.block_size,
        limit_samples=args.num_verify_stories,
        pre_tokenize=True,
    )

    x, y = dataset[0]
    print(f"Dataset item [0] X shape: {x.shape} | Y shape: {y.shape}")
    print(f"X (first 10 tokens): {x[:10].tolist()}")
    print(f"Y (first 10 tokens): {y[:10].tolist()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
