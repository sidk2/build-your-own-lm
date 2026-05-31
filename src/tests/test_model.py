import torch
import pytest
from model.gpt import GPT
from model.transformer import MultiHeadAttention, TransformerBlock

def test_multi_head_attention_shapes():
    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 4
    
    mha = MultiHeadAttention(num_heads=num_heads, d_model=d_model, use_bias=True, masked=True)
    x = torch.randn(batch_size, seq_len, d_model)
    out = mha(x)
    
    assert out.shape == (batch_size, seq_len, d_model)

def test_transformer_block_shapes():
    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 4
    
    block = TransformerBlock(num_heads=num_heads, d_model=d_model, use_bias=True, masked=True)
    x = torch.randn(batch_size, seq_len, d_model)
    out = block(x)
    
    assert out.shape == (batch_size, seq_len, d_model)

def test_gpt_forward_pass():
    vocab_size = 100
    d_model = 64
    num_heads = 4
    num_layers = 2
    batch_size = 3
    seq_len = 20
    
    gpt = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        use_bias=True,
        masked=True
    )
    
    # Input indices
    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = gpt(idx)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is None
    
    # Input with targets
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = gpt(idx, targets=targets)
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0 # Scalar tensor loss

def test_causal_masking():
    # Test that early tokens are unaffected by future tokens
    vocab_size = 50
    d_model = 16
    num_heads = 2
    num_layers = 1
    
    gpt = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        use_bias=False,
        masked=True
    )
    gpt.eval()
    
    # Two inputs that are identical up to sequence length 3, but differ at sequence length 4
    idx1 = torch.tensor([[1, 2, 3, 4]])
    idx2 = torch.tensor([[1, 2, 3, 5]])
    
    with torch.no_grad():
        logits1, _ = gpt(idx1)
        logits2, _ = gpt(idx2)
        
    # The outputs for the first 3 tokens should be identical because of the causal mask
    assert torch.allclose(logits1[:, :3, :], logits2[:, :3, :], atol=1e-5)
    # The output for the 4th token can differ
    assert not torch.allclose(logits1[:, 3, :], logits2[:, 3, :], atol=1e-5)
