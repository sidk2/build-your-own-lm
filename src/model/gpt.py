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
