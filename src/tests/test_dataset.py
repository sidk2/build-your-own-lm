import os
import tempfile
import torch
import pytest
from model.tokenizer import BPETokenizer
from data.dataset import TinyStoriesDataset

CORPUS = [
    "Once upon a time, there was a little bird.",
    "The bird liked to fly over the trees.",
]

def test_tokenizer_save_load():
    # Train a tokenizer
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(CORPUS)
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        tokenizer.save(tmp_path)
        
        # Load into a new tokenizer
        loaded = BPETokenizer(vocab_size=50)
        loaded.load(tmp_path)
        
        # Verify vocabularies match
        assert loaded.vocab_size == tokenizer.vocab_size
        assert loaded.vocabulary == tokenizer.vocabulary
        assert loaded.rev_vocab == tokenizer.rev_vocab
        assert loaded.token_ids == tokenizer.token_ids
        assert loaded.merge_rank == tokenizer.merge_rank
        
        # Verify encoding/decoding behavior matches
        text = "there was a bird"
        assert loaded.encode(text) == tokenizer.encode(text)
        assert loaded.decode(loaded.encode(text)) == tokenizer.decode(tokenizer.encode(text))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_dataset_on_the_fly():
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(CORPUS)
    
    # Initialize dataset with a tiny subset (limit_samples=2)
    # This will load from HF and cache if not already done.
    dataset = TinyStoriesDataset(
        split="train",
        tokenizer=tokenizer,
        block_size=16,
        limit_samples=2,
        pre_tokenize=False
    )
    
    assert len(dataset) == 2
    x, y = dataset[0]
    
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (16,)
    assert y.shape == (16,)
    # Target y should be shifted by 1 relative to x
    assert torch.equal(y[:-1], x[1:])

def test_dataset_pre_tokenized():
    tokenizer = BPETokenizer(vocab_size=50)
    tokenizer.train(CORPUS)
    
    dataset = TinyStoriesDataset(
        split="train",
        tokenizer=tokenizer,
        block_size=16,
        limit_samples=2,
        pre_tokenize=True
    )
    
    # Pre-tokenized size will depend on total token length chunked into block_size
    assert len(dataset) > 0
    x, y = dataset[0]
    
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert torch.equal(y[:-1], x[1:])
