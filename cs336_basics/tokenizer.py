from functools import lru_cache
import pickle
import regex as re
from typing import Iterable, Iterator

@lru_cache(maxsize=4096)
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

    @lru_cache(4096)
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