"""Sanity checks for the manual backward passes.

Run from the project directory:  python test_gradients.py
"""
import numpy as np

from model import GPT
from loss import CrossEntropyLoss
from optimizer import AdamW


def build(seed=0, vocab=50, C=16, T=8, B=2, layers=2):
    np.random.seed(seed)
    model = GPT(num_layers=layers, vocab_size=vocab, C=C, context_len=T)
    x = np.random.randint(0, vocab, (B, T))
    y = np.random.randint(0, vocab, (B, T))
    return model, CrossEntropyLoss(), x, y


def test_numerical_gradients(samples=10, h=1e-5, tol=1e-4):
    """Compare every parameter's analytic gradient with central differences."""
    model, loss_fn, x, y = build()
    params = model.parameters()

    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())

    worst = 0.0
    for i, p in enumerate(params):
        value, grad = p["value"], p["grad"].copy()
        numeric, analytic = [], []
        for _ in range(samples):
            idx = tuple(np.random.randint(0, s) for s in value.shape)
            old = value[idx]
            value[idx] = old + h
            plus = loss_fn.forward(model.forward(x), y)
            value[idx] = old - h
            minus = loss_fn.forward(model.forward(x), y)
            value[idx] = old
            numeric.append((plus - minus) / (2 * h))
            analytic.append(grad[idx])
        numeric, analytic = np.array(numeric), np.array(analytic)
        # relative error over the sampled entries of this parameter array
        rel = np.linalg.norm(numeric - analytic) / max(1e-12, np.linalg.norm(numeric) + np.linalg.norm(analytic))
        worst = max(worst, rel)
        assert rel < tol, f"param {i} {value.shape}: relative error {rel:.2e}"
    print(f"numerical gradient check passed ({len(params)} parameter arrays, max rel err {worst:.1e})")


def test_optimizer_sees_gradients():
    """Every parameter array handed to the optimizer must receive a non-zero gradient."""
    model, loss_fn, x, y = build()
    params = model.parameters()
    loss_fn.forward(model.forward(x), y)
    model.backward(loss_fn.backward())
    for i, p in enumerate(params):
        assert np.any(p["grad"] != 0), f"param {i} {p['value'].shape} has an all-zero gradient"
    print(f"all {len(params)} parameter arrays receive gradients")


def test_overfit_single_batch(steps=300, target=0.1):
    """A correct model should memorize one tiny batch."""
    model, loss_fn, x, y = build(vocab=100, C=32, T=16, B=4)
    optimizer = AdamW(model.parameters())
    for _ in range(steps):
        loss = loss_fn.forward(model.forward(x), y)
        model.backward(loss_fn.backward())
        optimizer.step()
        optimizer.zero_grad()
    assert loss < target, f"failed to overfit one batch: loss {loss:.4f}"
    print(f"overfit single batch: loss {loss:.4f} after {steps} steps")


if __name__ == "__main__":
    test_optimizer_sees_gradients()
    test_numerical_gradients()
    test_overfit_single_batch()
