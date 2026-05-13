# bpe-lite ⚡️

**A lightweight Byte Pair Encoding (BPE) tokenizer built from scratch in Python.**

`bpe-lite` is a fast, minimal, and educational implementation of BPE. It is heavily inspired by the curriculum of **Stanford CS336 (Language Modeling from Scratch)**, specifically mirroring the structure and requirements of their foundational tokenizer assignment. 

This package is designed to be easily readable for those learning how Large Language Models (LLMs) process text, while still implementing algorithmic optimizations that make it practical for small-to-medium scale dataset tokenization.

## ✨ Features

- **Inverted Index Optimization:** Training uses an inverted index to track byte pairs, drastically speeding up the merge process compared to naive brute-force counting.
- **$O(1)$ Merge Lookups:** The inference class (`Tokenizer`) pre-computes merge ranks, avoiding $O(N)$ list lookups during encoding.
- **Special Token Support:** Safely isolates and preserves special tokens (like `<|endoftext|>`) during pre-tokenization.
- **Compression Artifacts:** Automatically calculates and reports the dataset compression ratio upon training completion.
- **Modern Packaging:** Built using the `src/` layout and `pyproject.toml` for clean, reliable `pip` installation.

## 📦 Installation

You can install `bpe-lite` directly via pip:

```bash
pip install bpe-lite
```

*(Note: Requires `regex` as its only external dependency).*

## 🚀 Usage

You can use `bpe-lite` either directly from your terminal using the built-in CLI, or programmatically within your Python scripts.

### 1. Command Line Interface (CLI)
After installing the package, the `bpe-train` command becomes globally available. You can use it to train a new tokenizer on a raw text file.

```bash
bpe-train \
  --input ./data/train.txt \
  --vocab-size 10000 \
  --special-tokens "<|endoftext|>" "<|pad|>" \
  --out-dir ./tokenizer_models
```

This will process `train.txt`, calculate the optimal merges, print the final compression ratio, and save `vocab.pkl` and `merges.pkl` to the specified output directory.

### 2. Python API

#### Training a Tokenizer
You can invoke the training logic directly from Python if you are working inside a script or Jupyter Notebook.

```python
from bpe_lite import train
import pickle

# Train the tokenizer
vocab, merges = train(
    input_path="./data/train.txt", 
    vocab_size=10000, 
    special_tokens=["<|endoftext|>"]
)

# Save the artifacts manually
with open("vocab.pkl", 'wb') as f:
    pickle.dump(vocab, f)
with open("merges.pkl", 'wb') as f:
    pickle.dump(merges, f)
```

#### Encoding and Decoding (Inference)
Use the `Tokenizer` class to load your trained vocabulary and encode/decode text.

```python
from bpe_lite.tokenizer import Tokenizer

# Initialize from saved files
tokenizer = Tokenizer.from_files(
    vocab_filepath="vocab.pkl", 
    merges_filepath="merges.pkl", 
    special_tokens=["<|endoftext|>"]
)

# Encode raw text to integer IDs
text = "Hello world! <|endoftext|>"
ids = tokenizer.encode(text)
print(f"Token IDs: {ids}")

# Decode integer IDs back to strings
decoded_text = tokenizer.decode(ids)
print(f"Decoded text: {decoded_text}")
```

#### Lazy Encoding for Large Datasets
If you are processing massive datasets, you can use `encode_iterable` to lazily yield token IDs without blowing up your RAM:

```python
def text_stream():
    yield "First chunk of text."
    yield "Second chunk of text."

# Yields token IDs one by one
for token_id in tokenizer.encode_iterable(text_stream()):
    print(token_id)
```

## 📚 Acknowledgments
This repository was built as an educational exercise inspired by **Stanford CS336: Language Modeling from Scratch**. It serves as a practical demonstration of how modern LLM tokenizers (like OpenAI's `tiktoken`) operate under the hood.

## 🗺 Roadmap
- [x] Initial BPE implementation (CS336-inspired)
- [x] PyPI packaging and CLI
- [ ] Add unit tests (pytest) for round-trip encoding and special tokens
- [ ] Add support for custom regex patterns

## 📄 License
This project is licensed under the [MIT License](LICENSE).
