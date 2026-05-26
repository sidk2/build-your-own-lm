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

    # 1. Load subset of dataset for training the BPE tokenizer
    print("Step 1: Loading raw training stories from Hugging Face...")
    raw_dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Take a small subset for BPE training (since pure-Python BPE is O(num_merges * N))
    train_subset = raw_dataset.select(
        range(min(args.num_train_stories, len(raw_dataset)))
    )
    corpus = [item["text"] for item in train_subset]

    # 2. Initialize and Train BPETokenizer
    print("\nStep 2: Training BPETokenizer (this may take a few seconds)...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus)
    print(f"Training complete! Final vocabulary size: {len(tokenizer.vocabulary)}")

    # 3. Save the trained tokenizer
    print(f"\nStep 3: Saving tokenizer to '{args.tokenizer_path}'...")
    tokenizer.save(args.tokenizer_path)

    # 4. Verify deserialization
    print("\nStep 4: Verifying serialization/deserialization...")
    loaded_tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    loaded_tokenizer.load(args.tokenizer_path)
    print(
        f"Loaded tokenizer successfully. Vocab size: {len(loaded_tokenizer.vocabulary)}"
    )

    # 5. Build and verify TinyStoriesDataset
    print("\nStep 5: Initializing TinyStoriesDataset with loaded tokenizer...")
    dataset = TinyStoriesDataset(
        split="train",
        tokenizer=loaded_tokenizer,
        block_size=args.block_size,
        limit_samples=args.num_verify_stories,
        pre_tokenize=True,
    )

    print(f"Dataset block count: {len(dataset)}")

    # 6. Verify encoding and decoding
    print("\nStep 6: Running encoding/decoding verification on sample stories:")
    print("-" * 60)
    for idx in range(min(args.num_verify_stories, len(corpus))):
        original_text = corpus[idx]
        encoded_ids = loaded_tokenizer.encode(original_text)
        decoded_text = loaded_tokenizer.decode(encoded_ids)

        print(f"\nStory #{idx + 1} (Original Snippet):")
        print(f"  {original_text[:120].strip()}...")
        print(f"Tokenized IDs (first 20):")
        print(f"  {encoded_ids[:20]}")
        print(f"Token count: {len(encoded_ids)}")
        print(f"Decoded (first 120 chars):")
        print(f"  {decoded_text[:120].strip()}...")

        words_original = len(original_text.split())
        print(
            f"Word count: {words_original} | Token count: {len(encoded_ids)} | Compression Ratio: {words_original / len(encoded_ids):.2f}"
        )
        print("-" * 30)

    # Show a sample batch from PyTorch Dataset
    print("\nStep 7: Testing PyTorch Dataset __getitem__...")
    x, y = dataset[0]
    print(f"Dataset item [0] X shape: {x.shape} | Y shape: {y.shape}")
    print(f"X (first 10 tokens): {x[:10].tolist()}")
    print(f"Y (first 10 tokens): {y[:10].tolist()}")
    print("=" * 60)
    print("All tasks completed successfully!")


if __name__ == "__main__":
    main()
