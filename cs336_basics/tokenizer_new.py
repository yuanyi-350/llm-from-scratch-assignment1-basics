from functools import lru_cache
import pickle
import regex as re
from typing import Iterable, Iterator
from array import array

# @lru_cache(maxsize=1024)
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


GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

GPT2_RE = re.compile(GPT2_SPLIT_PATTERN)

def iter_pretokenize(text: str) -> Iterator[bytes]:
    """按 GPT-2 正则逐个产生字节串，零内存列表。"""
    for m in GPT2_RE.finditer(text):
        yield m.group(0).encode("utf-8")

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab: dict[int, bytes] = vocab
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] | None = special_tokens or []
        self.special_tokens_set: set[str] = set(special_tokens) if special_tokens else set()
        self.vocab_id: dict[bytes, int] = {value:key for key, value in vocab.items()}
        self.merges_id = {merge: idx for idx, merge in enumerate(merges)}
        self.pair_rank = {pair: k for k, pair in enumerate(merges)}
        self.pair2new = {(p1, p2): self.vocab_id[p1 + p2] for (p1, p2) in self.merges}

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

    def _encode_ordinary_text(self, text_bytes: bytes) -> list[int]:
        """BPE encode (不含特殊 token) —— 无额外列表 / O(n) 内存"""
        if not text_bytes:
            return []

        # ➊ 只解一次字节 → str
        try:
            text = text_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = text_bytes.decode("utf-8", errors="replace")

        ids_out = array("H")  # uint16 足够 ≤ 65k vocab

        for word_b in iter_pretokenize(text):
            # a. 初始：单字节 ids
            token_ids = array("H", (self.vocab_id[bytes([b])] for b in word_b))

            # b. 就地合并：最经典 “greedy smallest-rank merge until稳定”
            while True:
                best_rank = 1000000000
                best_pos = -1
                # ——— 找当前序列里 rank 最小的 pair ———
                for i in range(len(token_ids) - 1):
                    r = self.pair_rank.get(
                        (self.vocab[token_ids[i]], self.vocab[token_ids[i + 1]]),
                        1000000000,
                    )
                    if r < best_rank:
                        best_rank, best_pos = r, i
                if best_pos == -1:
                    break
                # ——— 替换 best_pos & best_pos+1 为新的 token ———
                new_id = self.pair2new[
                    (self.vocab[token_ids[best_pos]], self.vocab[token_ids[best_pos + 1]])
                ]
                token_ids[best_pos : best_pos + 2] = array("H", [new_id])

            ids_out.extend(token_ids)

        # ➌ array → Python list（评测期望 list）
        return ids_out.tolist()

    def encode(self, text: str) -> list[int]:
        """Encode str"""
        if not text:
            return []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if not sorted_special_tokens:
            return self._encode_ordinary_text(text.encode("utf-8"))

        special_pattern = f"({'|'.join(re.escape(s) for s in sorted_special_tokens)})"
        text_parts = re.split(special_pattern, text)

        all_ids = []
        for part in text_parts:
            if part in self.special_tokens:
                all_ids.append(self.vocab_id[part.encode("utf-8")])
            elif part:
                all_ids.extend(self._encode_ordinary_text(part.encode("utf-8")))
        return all_ids

    def encode_iterable(
        self,
        iterable: Iterable[str],
        *,
        output_format: str = "flat",
    ) -> Iterator[int] | Iterator[list[int]]:
        flat = output_format == "flat"
        for line in iterable:
            # —— 不要 strip 换行 ——          ▼
            ids = self.encode(line)
            if flat:
                yield from ids
            else:
                yield ids

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        byte_list = b''.join(self.vocab[k] for k in ids)
        return byte_list.decode('UTF-8', errors='replace')
