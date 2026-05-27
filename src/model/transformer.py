import torch
import torch.nn as nn

import math

    
# Note to self: another neat experiment could be comparing QK-Norm
# to scaled dot product attention

def sinusoidal_positional_embedding(seq_len : int, d_model : int, n : int = 10_000):
    pe = torch.zeros(seq_len, d_model)
    positions = torch.arange(0, seq_len).unsqueeze(1)
    
    # More numerically stable than pos / n^{2i / d_{model}}
    div_term = torch.exp(torch.arange(0, d_model, 2) * -math.log(n) / d_model)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term)
    return pe

class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_query_dim : int, value_dim : int, masked : bool):
        self.kq_dim = key_query_dim
        self.value_dim = value_dim
        self.masked = masked

        # In "Attention is All You Need", no linear layers in SDPA.
        # Just in MHA. Why?

        self.q_proj = nn.Linear(self.kq_dim, self.kq_dim, bias=False)
        self.k_proj = nn.Linear(self.kq_dim, self.kq_dim, bias=False)
        self.v_proj = nn.Linear(self.value_dim, self.value_dim, bias=False)

    def forward(self, query, key, value, mask):
        attn_scores = self.q_proj(query) @ self.k_proj(key).T / math.sqrt(self.kq_dim)
        if self.masked:
            attn_scores = torch.triu(attn_scores, diagonal=1)
        attn_probs = nn.functional.softmax(attn_scores, dim=-1) 
        return self.v_proj(attn_probs @ value)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads : int, d_model : int, use_bias : bool, masked : bool):
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        self.masked = masked
        
        self.heads = nn.ModuleList([
            ScaledDotProductAttention(key_query_dim=self.head_dim, value_dim=self.head_dim, masked=masked)
            for _ in range(num_heads)
        ])

        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)

        
    def forward(self, query, key, value, mask):
        values = torch.cat([h(query, key, value, mask) for h in self.heads], dim=2)
        return self.out_proj(values)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass


class TransformerBlock(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
