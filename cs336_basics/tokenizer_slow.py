import os
import multiprocessing
import time
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.tokenizer import _init_vocab, pre_tokenization, word_2_byte, _pretoken_worker_wrapper

def train_bpe_slow(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    num_processes = kwargs.get('num_processes')
    verbose = kwargs.get('verbose', False)
    vocab = _init_vocab({}, special_tokens)

    if verbose:
        start_time = time.time()

    if not num_processes:
        with open(input_path, 'r', encoding='UTF-8') as f:
            text = f.read()
        chunked_text = pre_tokenization(text, special_tokens)
        cnt_pretokens = Counter(map(word_2_byte, chunked_text))
        for st in special_tokens:
            del cnt_pretokens[word_2_byte(st)]
    else:
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((input_path, start, end, special_tokens))
        cnt_pretokens = Counter()

        with multiprocessing.Pool(processes=num_processes) as pool:
            for local_cnt in pool.imap_unordered(_pretoken_worker_wrapper, tasks, chunksize=1):
                cnt_pretokens.update(local_cnt)
    if verbose:
        count_duration = time.time() - start_time
        unique_pretokens = len(cnt_pretokens)
        total_pretokens = sum(cnt_pretokens.values())
        print(f"Pre-tokenization complete in {count_duration:.2f}s")
        print(f"Statistics: {unique_pretokens} unique pre-tokens, {total_pretokens} total tokens.")

    merges = []
    words = [[list(k), v] for k, v in cnt_pretokens.items()]
    token_to_idx = defaultdict(set)
    pair_counts = Counter()

    for idx, (tokens, count) in enumerate(words):
        for i in range(len(tokens) - 1):
            pair_counts[(tokens[i], tokens[i + 1])] += count
        for token in tokens:
            token_to_idx[token].add(idx)

    for n in range(len(vocab), vocab_size):
        if verbose and n % 500 == 0:
            count_duration = time.time() - start_time
            print(f"Merging token {n}/{vocab_size} | Time: {count_duration:.2f}s")

        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        if pair_counts[best_pair] <= 0:
            del pair_counts[best_pair]
            continue

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[n] = new_token

        p0, p1 = best_pair
        affected_indices = token_to_idx[p0] & token_to_idx[p1]

        for idx in affected_indices:
            tokens, count = words[idx]
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == p0 and tokens[i + 1] == p1:
                    if i > 0:
                        pair_counts[(tokens[i - 1], p0)] -= count
                    if i < len(tokens) - 2:
                        pair_counts[(p1, tokens[i + 2])] -= count
                    pair_counts[best_pair] -= count
                    tokens[i] = new_token
                    del tokens[i + 1]
                    if i > 0:
                        pair_counts[(tokens[i - 1], new_token)] += count
                    if i < len(tokens) - 1:
                        pair_counts[(new_token, tokens[i + 1])] += count
                    token_to_idx[new_token].add(idx)
                else:
                    i += 1
        del pair_counts[best_pair]
    return vocab, merges