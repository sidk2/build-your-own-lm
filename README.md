### Mini-GPT

## Motivation
My ML experience has largely skirted around language modeling. I come from more of a graph modeling/diffusion background, and my LLM experience is largely limited to applying pre-trained models and building agentic workflows. This repository is an attempt to fill in some of those gaps in my knowledge. 

## Implementation Plan

The goal of this repository is to build a small language model from scratch, training it on a small dataset to achieve reasonable performance. The model will be implemented in MLX. I will use the TinyStories dataset as the training data.

I will implement the tokenizer, embeddings, transformer blocks, and training loop. Afterwards, I will potentially then implement a mixture-of-experts version of the model, as I think that would present a fun challenge. 