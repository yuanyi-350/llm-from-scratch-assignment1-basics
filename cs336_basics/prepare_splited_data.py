from collections import Counter
import regex as re
import multiprocessing
import pickle
from pathlib import Path
from cs336_basics.pretokenization_example import find_chunk_boundaries

file_path = "/home/yuanyi/data/cs336/TinyStories/TinyStories-train.txt"

GPT4_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
pat = re.compile(GPT4_SPLIT_PATTERN)

def process_data_chunk(args):
    start, end = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)
        text = chunk_bytes.decode("utf-8", errors="ignore")
    tokens = pat.findall(text)
    tokens_bytes = [t.encode("utf-8") for t in tokens]
    return Counter(tokens_bytes)

with open(file_path, "rb") as f:
    num_processes = 64
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    print(f"✅ 文件已切分为 {len(boundaries) - 1} 个独立块")

tasks = []
for i in range(len(boundaries) - 1):
    tasks.append((boundaries[i], boundaries[i + 1]))

num_workers = 8
print(f"🔥 启动 {num_workers} 个进程进行并行计算...")

with multiprocessing.Pool(processes=num_workers) as pool:
    chunk_counters = pool.map(process_data_chunk, tasks)

global_counter = sum(chunk_counters, Counter())
output_path = Path(file_path).name + "vocab_counts.pkl"
with open(output_path, "wb") as f:
    pickle.dump(global_counter, f)
print(f"已保存到 {output_path}")
