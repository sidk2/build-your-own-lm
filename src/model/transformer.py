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

class MoEGate(nn.Module):
    def __init__(self, num_experts: int, n_selected: int, d_model: int, gate_noise_var: float = 1.0):
        super().__init__()

        assert num_experts >= n_selected, "Top-k for selection cannot exceed number of experts"
        self.relevance_model = nn.Linear(d_model, num_experts)
        self.noise_model = nn.Linear(d_model, num_experts)
        self.num_experts = num_experts
        self.k = n_selected
        self.d_model = d_model
        self.gate_noise_var = gate_noise_var
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor):
        relevance_scores = self.relevance_model(x)
        noise = torch.randn_like(relevance_scores) * self.gate_noise_var
        gated_logits = relevance_scores + noise * self.activation(self.noise_model(x))

        if self.k < self.num_experts:
            topk_vals, topk_idxs = torch.topk(gated_logits, self.k, dim=-1)
            gated = torch.full_like(gated_logits, float("-inf"))
            gated.scatter_(-1, topk_idxs, topk_vals)
        else:
            gated = gated_logits

        return nn.functional.softmax(gated, dim=-1)


class MoELayer(nn.Module):
    def __init__(self, num_experts: int, n_selected: int, d_model: int, gate_noise_var: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.k = n_selected
        self.d_model = d_model
        self.gate_noise_var = gate_noise_var

        self.gate = MoEGate(num_experts, n_selected, d_model, gate_noise_var)
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        gate_scores = self.gate(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        return torch.sum(gate_scores.unsqueeze(-1) * expert_outputs, dim=-2)


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        output = self.lin1(x)
        swish = output * torch.sigmoid(output)
        swiglu = swish * self.lin2(x)

        return swiglu


class ScaledDotProductAttention(nn.Module):
    def __init__(self, masked: bool, dropout: float = 0.1):
        super().__init__()
        self.masked = masked
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        elif self.masked:
            seq_len = query.size(-2)
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=query.device)
            ).view(1, 1, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_probs = nn.functional.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        return attn_probs @ value


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        use_bias: bool,
        masked: bool,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=use_bias)
        self.attn = ScaledDotProductAttention(masked=masked, dropout=dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.resid_dropout = nn.Dropout(dropout)

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

        return self.resid_dropout(self.out_proj(out))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

    def forward(self):
        pass


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        use_bias: bool,
        masked: bool,
        dropout: float = 0.1,
<<<<<<< HEAD
=======
        use_moe: bool = False,
        num_experts: int = 1,
        moe_top_k: int = 1,
        gate_noise_var: float = 1.0,
>>>>>>> 7a69064 (implemented mixture of experts with top-k routing)
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(
            num_heads=num_heads,
            d_model=d_model,
            use_bias=use_bias,
            masked=masked,
            dropout=dropout,
        )
        self.ln2 = nn.LayerNorm(d_model)

        if use_moe and num_experts > 1:
            self.mlp = MoELayer(
                num_experts=num_experts,
                n_selected=min(moe_top_k, num_experts),
                d_model=d_model,
                gate_noise_var=gate_noise_var,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model, bias=use_bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model, bias=use_bias),
                nn.Dropout(dropout),
            )

    def forward(self, x, mask=None):
        # Pre-LN and attention
        x = x + self.mha(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x
