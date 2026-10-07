# GPT from Scratch in NumPy

An educational GPT-style language model built with Python and NumPy. I built this project to understand how a transformer learns to predict the next token, from byte-level tokenization to the gradients used to update its weights.

The neural network, backward passes, and AdamW optimizer are implemented manually, without PyTorch, TensorFlow, or automatic differentiation. The project supports training, validation, and autoregressive text generation on CPU.

**Current result:** the recorded experiment shows decreasing training loss, but generated text is not yet consistently coherent. The notebook includes the training logs and generated samples.

## What is implemented

- **Byte-level BPE tokenizer:** starts with 256 byte tokens, learns frequent pair merges, and encodes and decodes text.
- **Embeddings:** learned token and positional embeddings.
- **Causal self-attention:** a single attention head per block, with query, key, value, and output projections.
- **Transformer blocks:** pre-layer normalization, residual connections, and a feed-forward network with ReLU and a hidden width of four times the embedding dimension.
- **Language-model output:** final layer normalization and a projection to vocabulary scores.
- **Manual backpropagation:** backward methods connecting the output layer to both embedding tables.
- **Training:** next-token batches, cross-entropy loss, AdamW updates, and validation loss monitoring.
- **Generation:** temperature-controlled sampling using the most recent tokens that fit within the context window.

## Architecture

The model takes a batch of token IDs and returns a vocabulary score for every position:

```text
Token IDs
    |
Token embeddings + positional embeddings
    |
Transformer block x N
    |-- LayerNorm -> causal self-attention -> residual addition
    |-- LayerNorm -> feed-forward network  -> residual addition
    |
Final LayerNorm
    |
Vocabulary projection
    |
Logits
```

Inputs have shape **(batch, sequence length)**. Logits have shape **(batch, sequence length, vocabulary size)**. During training, each position predicts the following token. During generation, the last position's logits determine the next sampled token.

## Project structure

| File | Purpose |
| --- | --- |
| [tokenizer.py](tokenizer.py) | Byte-level BPE training, encoding, and decoding |
| [data.py](data.py) | Random context windows and shifted next-token targets |
| [embedding.py](embedding.py) | Token and positional embeddings |
| [linear.py](linear.py) | Linear layers and their backward passes |
| [activation.py](activation.py) | Softmax and ReLU |
| [layernorm.py](layernorm.py) | Layer normalization and its backward pass |
| [attention.py](attention.py) | Causal self-attention |
| [feedforward.py](feedforward.py) | Feed-forward network |
| [transformer.py](transformer.py) | Transformer block composition |
| [model.py](model.py) | Complete GPT model, backpropagation, parameters, and generation |
| [loss.py](loss.py) | Cross-entropy loss and its backward pass |
| [optimizer.py](optimizer.py) | AdamW updates and gradient resetting |
| [train.ipynb](train.ipynb) | Data preparation, training, validation, and generation experiments |
| [requirements.txt](requirements.txt) | Packages needed to run the notebook |

## Getting started

### 1. Get the code and install dependencies

Clone the repository, enter its directory, and install the dependencies into the Python environment you will use for the notebook:

```bash
git clone https://github.com/BerkinOzcan88/small-gpt.git
cd small-gpt
python -m pip install -r requirements.txt
```

If you already have a local checkout, start from its project directory instead. The folder name can be different; the code does not depend on it.

### 2. Download the dataset

The dataset is **[musabg/wikipedia-tr](https://huggingface.co/datasets/musabg/wikipedia-tr)** on Hugging Face. Its dataset card calls it **Turkish Wikipedia, May 2023**.

Create a `wikipedia-tr/data/` directory inside the project and download the first shard below, preserving its original filename:

| Original filename | Needed by this notebook? |
| --- | --- |
| [train-00000-of-00002-ed6b025df7a1f653.parquet](https://huggingface.co/datasets/musabg/wikipedia-tr/resolve/main/data/train-00000-of-00002-ed6b025df7a1f653.parquet?download=true) | Yes; approximately 329 MB |
| [train-00001-of-00002-0aa63953f8b51c17.parquet](https://huggingface.co/datasets/musabg/wikipedia-tr/resolve/main/data/train-00001-of-00002-0aa63953f8b51c17.parquet?download=true) | No; the second shard is optional |

The expected layout is:

```text
<project directory>/
├── train.ipynb
├── requirements.txt
└── wikipedia-tr/
    └── data/
        └── train-00000-of-00002-ed6b025df7a1f653.parquet
```

`DATA_PATH` uses this relative path, so no username or machine-specific directory needs to be changed. If you keep the data elsewhere, set `DATA_PATH` to that Parquet file in the configuration cell.

### 3. Open the training notebook

From the project directory, start JupyterLab:

```bash
python -m jupyterlab
```

Open [train.ipynb](train.ipynb) and run its cells in order:

1. Import the modules and set the model configuration and dataset path.
2. Load the text and train the tokenizer.
3. Split the token sequence into training and validation portions.
4. Initialize the model, loss, and optimizer.
5. Run the training loop and inspect losses and generated samples.

The training loop is configured for **15,000 steps**. Set a smaller step count before running that cell if you only want to check the workflow.

### 4. Generate text

Generation is included in the training loop. The notebook uses the prompt `istanbul`, samples 100 additional tokens, and uses a temperature of 0.7.

To use another prompt, change the text passed to the tokenizer when `start` is created. The model uses a sliding context window as the generated sequence grows.

The current workflow keeps the trained model and tokenizer in the notebook session; it does not include checkpoint saving and loading.

## Recorded experiment

The saved notebook uses the first **50 articles** from the original first shard, `train-00000-of-00002-ed6b025df7a1f653.parquet`. Its outputs report **1,102,264 characters** and **284,345 tokens** after BPE training.

| Setting | Value |
| --- | --- |
| Transformer blocks | 4 |
| Attention heads per block | 1 |
| Embedding dimension | 128 |
| Feed-forward hidden dimension | 512 |
| Context length | 64 tokens |
| Vocabulary size | 5,000 |
| Batch size | 16 |
| Training / validation split | First 90% / final 10% of the token sequence |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Weight decay | 0.01 |

The tokenizer is fitted on the selected text before the token sequence is split. Validation loss is measured on one randomly sampled batch every 50 steps, so these are sampled batch measurements rather than full validation-set averages.

The saved output reaches step **9,000**. Selected recorded cross-entropy losses are:

| Logged step | Training loss | Validation loss |
| --- | ---: | ---: |
| 0 | 9.5406 | 9.5437 |
| 9,000 | 6.1492 | 8.1184 |
| 9,050 | 5.9503 | 8.3368 |

Training loss decreases substantially, while validation loss remains higher. This suggests limited generalization in this experiment. Generated samples contain recognizable Turkish fragments, but do not reliably form coherent sentences. Full samples are preserved in the notebook. The updated plotting cell places validation measurements at their actual training steps; its old saved plot has been cleared.

## Scope and limitations

This project focuses on understanding and implementing a language model end to end.

- Standard NumPy runs the model on CPU, which limits practical training throughput.
- The documented experiment uses a small dataset subset and a 64-token context.
- The implementation uses single-head attention.
- Output quality depends on data, optimization, capacity, and implementation correctness as well as training time.
- The repository does not yet include an automated numerical gradient-check suite or a documented tiny-batch overfitting test. Decreasing loss alone does not establish the correctness of every backward calculation.

## Learning focus

The project covers how causal masking enforces next-token prediction, how gradients pass through attention and residual connections, and how parameter and gradient arrays are shared with an optimizer. It also provides a practical way to examine the gap between training loss, validation loss, and the quality of generated text.
