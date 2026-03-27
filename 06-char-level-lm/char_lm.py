"""Tiny character-level language model.

This is the next curriculum step after MNIST MLP: learning to model sequences.
It implements a simple bigram model in PyTorch and samples synthetic names.

Run:
    python 06-char-level-lm/char_lm.py
"""

from pathlib import Path
import torch
import torch.nn.functional as F

DATA_PATH = Path(__file__).with_name("names.txt")
SEED = 42


def load_words(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_vocab(words: list[str]):
    chars = sorted(list(set("".join(words))))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def build_dataset(words: list[str], stoi: dict[str, int]):
    xs, ys = [], []
    for word in words:
        chs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


def train_bigram(xs: torch.Tensor, ys: torch.Tensor, vocab_size: int, steps: int = 200, lr: float = 30.0):
    g = torch.Generator().manual_seed(SEED)
    W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    for step in range(steps):
        xenc = F.one_hot(xs, num_classes=vocab_size).float()
        logits = xenc @ W
        loss = F.cross_entropy(logits, ys)

        W.grad = None
        loss.backward()
        W.data += -lr * W.grad

        if step % 50 == 0 or step == steps - 1:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    return W


def sample_names(W: torch.Tensor, itos: dict[int, str], n: int = 10):
    g = torch.Generator().manual_seed(SEED)
    vocab_size = W.shape[0]

    for _ in range(n):
        out = []
        ix = 0
        while True:
            xenc = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
            logits = xenc @ W
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            if ix == 0:
                break
            out.append(itos[ix])
        print("".join(out))


def main():
    words = load_words(DATA_PATH)
    stoi, itos = build_vocab(words)
    xs, ys = build_dataset(words, stoi)
    vocab_size = len(stoi)

    print(f"Loaded {len(words)} words")
    print(f"Vocab size: {vocab_size}")
    print(f"Training examples: {len(xs)}")

    W = train_bigram(xs, ys, vocab_size=vocab_size)

    print("\nGenerated samples:")
    sample_names(W, itos)


if __name__ == "__main__":
    main()
