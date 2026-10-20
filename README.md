# GPT from Scratch in NumPy

A small GPT-style language model written in plain Python and NumPy. The forward pass, backward pass and AdamW optimizer are all implemented by hand, with no PyTorch, TensorFlow or autograd. It trains and generates text on CPU.

## What's inside

- Byte-level BPE tokenizer
- Token and positional embeddings
- Causal self-attention (one head per block)
- Transformer blocks with pre-LayerNorm, residual connections and a ReLU feed-forward network
- Manual backpropagation through the whole model
- Cross-entropy loss and AdamW
- Temperature-based text generation

```text
Token IDs -> embeddings -> Transformer block x N -> LayerNorm -> vocabulary projection -> logits
```

## Files

| File | Purpose |
| --- | --- |
| [tokenizer.py](tokenizer.py) | BPE training, encoding and decoding |
| [data.py](data.py) | Random training batches |
| [embedding.py](embedding.py) | Token and positional embeddings |
| [linear.py](linear.py) | Linear layer |
| [activation.py](activation.py) | Softmax and ReLU |
| [layernorm.py](layernorm.py) | Layer normalization |
| [attention.py](attention.py) | Causal self-attention |
| [feedforward.py](feedforward.py) | Feed-forward network |
| [transformer.py](transformer.py) | Transformer block |
| [model.py](model.py) | Full GPT model and generation |
| [loss.py](loss.py) | Cross-entropy loss |
| [optimizer.py](optimizer.py) | AdamW |
| [test_gradients.py](test_gradients.py) | Gradient tests |
| [train.ipynb](train.ipynb) | Training and generation notebook |


## Default configuration

| Setting | Value |
| --- | --- |
| Transformer blocks | 4 |
| Embedding dimension | 128 |
| Context length | 64 tokens |
| Vocabulary size | 1,000 |
| Batch size | 16 |
| Learning rate | 0.001 |
| Weight decay | 0.01 |

## Limitations

- Runs on CPU only, so training is slow.
- Single-head attention.
- No checkpoint saving or loading.
