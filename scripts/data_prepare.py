"""
data_prepare.py

```bash
uv run --no-project --python pypy --with regex python -m scripts.data_prepare\
 --input_path "/home/yuanyi/data/cs336/TinyStories/TinyStoriesV2-GPT4-train.txt"
```
"""
import argparse
import pickle
from cs336_basics.tokenizer import train_bpe
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to raw text file")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save .bin files")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocab size for training tokenizer")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    vocab_size = args.vocab_size

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = input_path.stem.lower()
    vocab_path = output_dir / f"{dataset_name}_vocab.pkl"
    merges_path = output_dir / f"{dataset_name}_merges.pkl"

    print(f"Training new Tokenizer with vocab_size={vocab_size}...")
    vocab, merges = train_bpe(input_path, vocab_size, ["<|endoftext|>"], num_processes=16, verbose=True)

    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
        print(f"Saved new tokenizer to {vocab_path.absolute()}")
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
        print(f"Saved new tokenizer to {merges_path.absolute()}")
