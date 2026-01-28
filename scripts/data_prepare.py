"""
data_prepare.py

```bash
uv run --no-project --python pypy --with regex python -m cs336_basics.data_prepare\
 --input_path "/home/yuanyi/data/cs336/TinyStories/TinyStoriesV2-GPT4-train.txt"
```
"""
import os
import argparse
import pickle
from cs336_basics.tokenizer import train_bpe


def prepare(input_path, output_dir, vocab_size):
    filename = os.path.basename(input_path)
    dataset_name = os.path.splitext(filename)[0].lower()
    if "tinystories" in dataset_name:
        dataset_name = "tinystories"

    vocab_path = os.path.join(output_dir, f"{dataset_name}_vocab.pkl")
    merges_path = os.path.join(output_dir, f"{dataset_name}_merges.pkl")

    print(f"Training new Tokenizer with vocab_size={vocab_size}...")
    vocab, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"], num_processes=16, verbose=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    print(f"Saved new tokenizer to {vocab_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw text file")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save .bin files")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocab size for training tokenizer")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prepare(args.input_path, args.output_dir, args.vocab_size)