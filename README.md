### Mini-GPT

## Motivation
My ML experience has largely skirted around language modeling. I come from more of a graph modeling/diffusion background, and my LLM experience is largely limited to applying pre-trained models and building agentic workflows. This repository is an attempt to fill in some of those gaps in my knowledge. 

## Implementation Plan

The goal of this repository is to build a small language model from scratch, training it on a small dataset to achieve reasonable performance. I will use the TinyStories dataset as the training data.

I will implement the tokenizer, embeddings, transformer blocks, and training loop. Afterwards, I will potentially then implement a mixture-of-experts version of the model, as I think that would present a fun challenge. 

## Running

Install dependencies:

```bash
uv sync
```

Train the BPE tokenizer on TinyStories:

```bash
uv run mini-gpt-tokenizer \
  --vocab-size 50000 \
  --num-train-stories 50000 \
  --tokenizer-path data/tokenizer.json
```

Train the GPT model:

```bash
uv run mini-gpt-train \
  --tokenizer-path data/tokenizer.json \
  --out-dir data/out \
  --context-length 1024 \
  --batch-size 32 \
  --max-iters 5000
```

For a smaller smoke test:

```bash
uv run mini-gpt-tokenizer \
  --vocab-size 1000 \
  --num-train-stories 1000 \
  --num-verify-stories 10 \
  --tokenizer-path data/tokenizer.json

uv run mini-gpt-train \
  --tokenizer-path data/tokenizer.json \
  --out-dir data/out \
  --context-length 128 \
  --batch-size 4 \
  --max-iters 20 \
  --eval-interval 10 \
  --eval-iters 2 \
  --limit-samples 100
```

The tokenizer is saved to `data/tokenizer.json` by default, and training writes `ckpt_latest.pt` and `ckpt_best.pt` under `data/out`.

## Notes on Experiments
Planned experiments:
    - Efficiency of BPE tokenizer with various optimizations (low priority)
    - Performance difference between RoPE and sinusoidal embeddings
    - Performance difference between bias and bias free linear layers in attention
