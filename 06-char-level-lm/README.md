# 06 — Character-Level Language Model

This stage extends the repo from feedforward networks into sequence modelling.

The goal is to build intuition for how language models work at the simplest possible level: predicting the next character from previous characters.

## What this section demonstrates

- turning text into a dataset for modelling
- building a character vocabulary and integer encoding
- learning next-token probabilities from context
- training a small neural language model in PyTorch
- generating text autoregressively from the learned distribution

## Why it matters

This is the bridge between:
- MNIST-style supervised learning
- and sequence models / transformers

It introduces the core language modelling loop: **context -> prediction -> sampling -> generation**.

## Included here

- `char_lm.py` — small character-level bigram language model
- `names.txt` — tiny demo corpus for local experimentation

## Run

```bash
python 06-char-level-lm/char_lm.py
```

The script:
1. loads the sample corpus
2. builds a character vocabulary
3. trains a simple bigram model
4. samples a few generated names

## Next likely extensions

- move from bigram probabilities to an MLP-based next-character model
- add embedding layers and hidden state
- build toward makemore-style word generation
- use this section as the conceptual runway for transformer fundamentals
