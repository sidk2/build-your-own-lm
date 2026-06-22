import torch
import pytest
from model.gpt import GPT
from model.transformer import MoEGate, MoELayer, MultiHeadAttention, TransformerBlock

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

def test_transformer_block_shapes_with_moe():
    batch_size = 2
    seq_len = 10
    d_model = 32
    num_heads = 4
    num_experts = 4
    moe_top_k = 2

    block = TransformerBlock(
        num_heads=num_heads,
        d_model=d_model,
        use_bias=True,
        masked=True,
        use_moe=True,
        num_experts=num_experts,
        moe_top_k=moe_top_k,
    )
    x = torch.randn(batch_size, seq_len, d_model)
    out = block(x)

    assert out.shape == (batch_size, seq_len, d_model)


def test_moe_gate_topk_sparsity():
    batch_size = 3
    seq_len = 5
    d_model = 16
    num_experts = 6
    top_k = 2

    gate = MoEGate(num_experts=num_experts, n_selected=top_k, d_model=d_model, gate_noise_var=0.0)
    x = torch.randn(batch_size, seq_len, d_model)
    gate_probs = gate(x)

    assert gate_probs.shape == (batch_size, seq_len, num_experts)
    selected_counts = (gate_probs > 0).sum(dim=-1)
    assert torch.all(selected_counts == top_k)
    assert torch.allclose(gate_probs.sum(dim=-1), torch.ones_like(gate_probs.sum(dim=-1)))


def test_moe_layer_output_aggregation():
    batch_size = 2
    seq_len = 4
    d_model = 16
    num_experts = 3
    top_k = 2

    layer = MoELayer(
        num_experts=num_experts,
        n_selected=top_k,
        d_model=d_model,
        gate_noise_var=0.0,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    out = layer(x)

    gate_probs = layer.gate(x)
    manual_outputs = torch.stack([expert(x) for expert in layer.experts], dim=-2)
    expected = torch.sum(gate_probs.unsqueeze(-1) * manual_outputs, dim=-2)

    assert out.shape == (batch_size, seq_len, d_model)
    assert torch.allclose(out, expected)


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


def test_gpt_forward_pass_with_moe():
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
        masked=True,
        use_moe=True,
        num_experts=4,
        moe_top_k=2,
        gate_noise_var=0.5,
    )

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = gpt(idx)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss is None

    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, loss = gpt(idx, targets=targets)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0

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


def test_generation_shape_and_range():
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
        masked=True,
    )
    gpt.eval()

    batch_size = 2
    seq_len = 5
    max_new_tokens = 10

    idx = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Test generation
    out = gpt.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10)

    # Expected output shape: [batch_size, seq_len + max_new_tokens]
    assert out.shape == (batch_size, seq_len + max_new_tokens)

    # Assert generated tokens are within vocabulary
    assert torch.all(out >= 0)
    assert torch.all(out < vocab_size)

    # Assert original sequence was preserved
    assert torch.equal(out[:, :seq_len], idx)

