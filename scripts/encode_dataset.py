import pickle
import time
from tqdm import tqdm
import argparse
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries

global_tokenizer = None


def init_worker(vocab_path, merges_path, special_tokens):
    global global_tokenizer
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(merges_path, 'rb') as f:
        merges = pickle.load(f)
    global_tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def count_tokens(task):
    path_to_txt, start_byte, end_byte = task
    with open(path_to_txt, 'rb') as f:
        f.seek(start_byte)
        chunk_data = f.read(end_byte - start_byte)
        text = chunk_data.decode("utf-8", errors="ignore")
        ids = global_tokenizer.encode(text)
        return len(ids)


def write_tokens(task):
    path_to_txt, save_path, start_byte, end_byte, mm_start_idx, dtype, total_shape = task
    with open(path_to_txt, 'rb') as f:
        f.seek(start_byte)
        chunk_data = f.read(end_byte - start_byte)
        text = chunk_data.decode("utf-8", errors="ignore")
        ids = global_tokenizer.encode(text)
    tokens_mm = np.memmap(save_path, dtype=dtype, mode='r+', shape=total_shape)
    tokens_mm[mm_start_idx: mm_start_idx + len(ids)] = ids
    tokens_mm.flush()
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="./data/owt_train.dat")
    parser.add_argument("--tokenizer_dir", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=64)
    args = parser.parse_args()

    start_time = time.time()

    vocab_path = Path(args.tokenizer_dir) / "owt_train_vocab.pkl"
    merges_path = Path(args.tokenizer_dir) / "owt_train_merges.pkl"
    abs_output_path = Path(args.output).absolute()
    abs_output_path.parent.mkdir(parents=True, exist_ok=True)
    special_tokens = ["<|endoftext|>"]
    dtype = np.int16

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_workers, b"<|endoftext|>")

    chunk_tasks = []
    for i in range(len(boundaries) - 1):
        chunk_tasks.append((args.input, boundaries[i], boundaries[i + 1]))

    with Pool(processes=args.num_workers, initializer=init_worker,
              initargs=(vocab_path, merges_path, special_tokens)) as pool:
        counts = list(tqdm(pool.imap(count_tokens, chunk_tasks), total=len(chunk_tasks), desc="Counting tokens"))

    total_tokens = sum(counts)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    tokens_mm_init = np.memmap(args.output, dtype=dtype, mode='w+', shape=(total_tokens,))
    del tokens_mm_init

    offsets = np.cumsum([0] + counts)[:-1]
    write_tasks = []
    for i in range(len(chunk_tasks)):
        write_tasks.append((
            args.input, args.output, chunk_tasks[i][1], chunk_tasks[i][2],
            offsets[i], dtype, (total_tokens,)
        ))

    with Pool(processes=args.num_workers, initializer=init_worker,
              initargs=(vocab_path, merges_path, special_tokens)) as pool:
        list(tqdm(pool.imap(write_tokens, write_tasks), total=len(write_tasks), desc="Writing tokens"))

    end_time = time.time()
    print(f"\n[Done] Total tokens: {total_tokens}")
    print(f"[Done] Saved to: {abs_output_path}")
    print(f"[Done] Total time: {end_time - start_time:.2f} seconds")