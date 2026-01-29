from functools import lru_cache
import pickle
import os
import multiprocessing
import time
import regex as re
from typing import Iterable, Iterator
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries

@lru_cache(maxsize=4096) # 评测机如果超内存就去掉这个
def word_2_byte(word: str) -> tuple[bytes, ...]:
    word_decoded = list(word.encode('UTF-8'))
    word_byte = [bytes([b]) for b in word_decoded]
    return tuple(word_byte)

def pre_tokenization(s: str, special_token: list[str]) -> list[str]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if not special_token:
        return re.findall(PAT, s)

    toks = sorted(special_token, key=len, reverse=True) # 长→短排序，防止短的抢先匹配
    union = "|".join(re.escape(t) for t in toks)
    parts = re.split(f"({union})", s)

    out = []
    st = set(special_token)
    for part in parts:
        if not part:
            continue
        if part in st:
            out.append(part)
        else:
            out.extend(re.findall(PAT, part))
    return out


def _pretoken_worker_wrapper(args):
    input_path, start, end, special_tokens = args
    local_counter = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    toks = sorted(special_tokens, key=len, reverse=True)
    union = "|".join(re.escape(t) for t in toks)
    splitter = re.compile(f"({union})")
    pat_regex = re.compile(PAT)
    special_tokens_set = set(special_tokens)

    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    text = data.decode("utf-8", errors="replace")
    parts = splitter.split(text)

    for part in parts:
        if not part:
            continue
        if part in special_tokens_set:
            continue
        for m in pat_regex.finditer(part):
            token = word_2_byte(m.group(0))
            local_counter[token] += 1
    return local_counter



class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens
        self.special_tokens_set: set[str] = set(special_tokens) if special_tokens else set()
        self.vocab_id: dict[bytes, int] = {value:key for key, value in vocab.items()}
        self.merges_id = {merge: idx for idx, merge in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Class method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:
        """
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)

        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)

        return cls(vocab, merges, special_tokens)

    @lru_cache(4096) # 评测机如果超内存就去掉这个
    def _encode_word(self, word : str) -> list[int] :
        word_byte_list = list(word_2_byte(word))

        while len(word_byte_list) > 1:
            word_pairs = set((word_byte_list[k], word_byte_list[k + 1])
                             for k in range(len(word_byte_list) - 1))
            if not word_pairs:
                break
            bigram = min(word_pairs, key=lambda pair: self.merges_id.get(pair, float('inf')))
            if bigram not in self.merges_id:
                break
            k = 0
            new_byte_token = []
            while k < len(word_byte_list):
                if k < len(word_byte_list) - 1 and (word_byte_list[k], word_byte_list[k + 1]) == bigram:
                    new_byte_token.append(bigram[0] + bigram[1])
                    k += 2
                else:
                    new_byte_token.append(word_byte_list[k])
                    k += 1
            word_byte_list = new_byte_token

        res = []
        for merged_bytes in word_byte_list:
            if merged_bytes in self.vocab_id:
                res.append(self.vocab_id[merged_bytes])
            else:
                print(f"\033[33mWarning: {merged_bytes} not in vocab\033[0m")
        return res

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.
        """
        res = []
        parts = pre_tokenization(text, self.special_tokens)
        for part in parts:
            if part in self.special_tokens_set:
                res.append(self.vocab_id[part.encode('utf-8')])
                continue
            res += self._encode_word(part)
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        byte_list = b''.join(self.vocab[k] for k in ids)
        return byte_list.decode('UTF-8', errors='replace')

def train_bpe(
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




def _init_vocab(vocab: dict, special_token: list):
    special_token_encoded = [s.encode('UTF-8') for s in special_token]
    idx = 0
    for code in special_token_encoded:
        vocab[idx] = code
        idx += 1

    for i in range(256):
        init_str = bytes([i])
        if init_str not in vocab.values():
            vocab[idx] = init_str
            idx += 1
    return vocab
