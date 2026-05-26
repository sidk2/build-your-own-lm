import os
from typing import List, Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
DATA_DIR = "data"

class TinyStoriesDataset(Dataset):
    """
    PyTorch Dataset for Hugging Face TinyStories dataset.
    """
    def __init__(
        self,
        split: str = "train",
        tokenizer = None,
        block_size: int = 256,
        limit_samples: Optional[int] = None,
        pre_tokenize: bool = True,
    ):
        """
        Args:
            split: 'train' or 'validation'
            tokenizer: An instance of BPETokenizer
            block_size: Context length for training (maximum sequence length)
            limit_samples: If set, limits the number of stories loaded (useful for fast testing/dev)
            pre_tokenize: If True, pre-tokenizes the entire dataset and chunks it. If False, tokenizes on-the-fly.
        """
        self.split = split
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pre_tokenize = pre_tokenize

        print(f"Loading TinyStories '{split}' split from Hugging Face...")
        # Note: the validation split on HF dataset might be named 'validation' or similar.
        # HF 'roneneldan/TinyStories' has 'train' and 'validation' splits.
        self.raw_dataset = load_dataset("roneneldan/TinyStories", split=split)

        # Limit samples if requested
        if limit_samples is not None:
            self.raw_dataset = self.raw_dataset.select(range(min(limit_samples, len(self.raw_dataset))))

        if self.pre_tokenize and self.tokenizer is not None:
            self._pretokenize_dataset()
        else:
            self.tokens_data = None

    def _pretokenize_dataset(self):
        print(f"Pre-tokenizing {len(self.raw_dataset)} stories...")
        all_tokens = []
        from tqdm import tqdm
        pad_id = self.tokenizer.rev_vocab.get("<pad>", 0)
        
        for item in tqdm(self.raw_dataset, desc="Tokenizing"):
            text = item["text"]
            ids = self.tokenizer.encode(text)
            # Add padding/EOS token to separate stories
            all_tokens.extend(ids + [pad_id])

        self.tokens_data = torch.tensor(all_tokens, dtype=torch.long)
        self.num_blocks = max(1, len(self.tokens_data) // self.block_size)

    def __len__(self) -> int:
        if self.pre_tokenize and self.tokens_data is not None:
            return self.num_blocks
        return len(self.raw_dataset)

    def __getitem__(self, idx: int):
        if self.pre_tokenize and self.tokens_data is not None:
            start = idx * self.block_size
            end = start + self.block_size
            
            if end + 1 <= len(self.tokens_data):
                chunk = self.tokens_data[start : end + 1]
                x = chunk[:-1]
                y = chunk[1:]
            else:
                # Handle edge case at the very end
                x = self.tokens_data[start:end]
                pad_id = self.tokenizer.rev_vocab.get("<pad>", 0)
                # Pad x if it's shorter than block_size
                if len(x) < self.block_size:
                    padding = torch.full((self.block_size - len(x),), pad_id, dtype=torch.long)
                    x = torch.cat([x, padding])
                
                # Create y shifted by 1
                y = torch.cat([self.tokens_data[start+1:end], torch.tensor([pad_id], dtype=torch.long)])
                if len(y) < self.block_size:
                    padding = torch.full((self.block_size - len(y),), pad_id, dtype=torch.long)
                    y = torch.cat([y, padding])
            return x, y
        else:
            # Tokenize on-the-fly
            text = self.raw_dataset[idx]["text"]
            pad_id = self.tokenizer.rev_vocab.get("<pad>", 0) if self.tokenizer is not None else 0
            ids = self.tokenizer.encode(text) if self.tokenizer is not None else [0]
            
            # Truncate or pad to block_size + 1
            if len(ids) > self.block_size + 1:
                ids = ids[: self.block_size + 1]
            else:
                ids = ids + [pad_id] * (self.block_size + 1 - len(ids))

            x = torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            return x, y