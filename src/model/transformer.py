import torch
import torch.nn as nn

import math

# Note to self: another neat experiment could be comparing QK-Norm
# to scaled dot product attention


def sinusoidal_positional_embedding(seq_len: int, d_model: int, n: int = 10_000):
    pe = torch.zeros(seq_len, d_model)
    positions = torch.arange(0, seq_len).unsqueeze(1)

    # More numerically stable than pos / n^{2i / d_{model}}
    div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(n) / d_model)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe


class ScaledDotProductAttention(nn.Module):
    def __init__(self, masked: bool):
        super().__init__()
        self.masked = masked

    def forward(self, query, key, value, mask=None):
        # query, key, value shapes: [batch_size, num_heads, seq_len, head_dim]
        # attn_scores shape: [batch_size, num_heads, seq_len, seq_len]
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        elif self.masked:
            seq_len = query.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device)).view(1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
            
        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        return attn_probs @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, use_bias: bool, masked: bool):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        # Single projection for Q, K, V. Is more efficient to use one 
        # large matrix and then chunk it, than to use three separate
        # matrices. 
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.attn = ScaledDotProductAttention(masked=masked)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        qkv = self.qkv_proj(x)
        
        # Split into Q, K, V: each is [batch_size, seq_len, d_model]
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape to [batch_size, seq_len, num_heads, head_dim] and transpose to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        out = self.attn(q, k, v, mask)

        # Reshape back to [batch_size, seq_len, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int, d_model: int, use_bias: bool, masked: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            use_bias=use_bias,
            masked=masked,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=use_bias),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=use_bias)
        )

    def forward(self, x, mask=None):
        # Pre-LN and attention
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x