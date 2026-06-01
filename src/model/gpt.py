import torch
import torch.nn as nn
from model import transformer as tfmr
from model import tokenizer as tkn


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        use_bias: bool,
        masked: bool,
        tokenizer_path: str = None,
        context_length: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.masked = masked
        self.context_length = context_length

        # Load tokenizer if path provided
        if tokenizer_path is not None:
            self.tokenizer = tkn.BPETokenizer(vocab_size=vocab_size)
            self.tokenizer.load(tokenizer_path)
        else:
            self.tokenizer = None

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # We pre-compute sinusoidal embeddings for a max context length of 512.
        # Registered as buffer so it moves with the model to GPU/CPU but is not trained.
        self.register_buffer(
            "pos_embedding",
            tfmr.sinusoidal_positional_embedding(
                seq_len=self.context_length, d_model=d_model
            ),
        )

        self.layers = nn.ModuleList(
            [
                tfmr.TransformerBlock(
                    num_heads=num_heads,
                    d_model=d_model,
                    use_bias=use_bias,
                    masked=masked,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=use_bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        # idx shape: [batch_size, seq_len]
        batch_size, seq_len = idx.size()

        token_emb = self.token_embedding(idx)  # [batch_size, seq_len, d_model]
        pos_emb = self.pos_embedding[:seq_len, :].unsqueeze(0)  # [1, seq_len, d_model]
        x = token_emb + pos_emb  # [batch_size, seq_len, d_model]

        mask = None
        if self.masked:
            mask = torch.tril(torch.ones((seq_len, seq_len), device=idx.device)).view(
                1, 1, seq_len, seq_len
            )

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch_size, seq_len, vocab_size]

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=(
                    self.tokenizer.rev_vocab.get("<pad>", -1) if self.tokenizer else -1
                ),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        Generate new tokens given a conditioning sequence of indices.
        idx: [batch_size, seq_len] tensor of token IDs
        """
        # Ensure we are in eval mode for generation
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            # Crop index to context length if needed
            idx_cond = idx[:, -self.context_length :]

            # Forward pass
            logits, _ = self(idx_cond)
            # Focus only on the last time step: shape [batch_size, vocab_size]
            logits = logits[:, -1, :]

            if temperature != 1.0 and temperature > 0.0:
                logits = logits / temperature

            # Optionally crop to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append sampled index
            idx = torch.cat((idx, idx_next), dim=1)

        if was_training:
            self.train()

        return idx

