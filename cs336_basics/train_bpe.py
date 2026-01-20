import pickle
from pathlib import Path
from collections import Counter

file_path = "/home/yuanyi/data/cs336/TinyStories/TinyStories-train.txt"
pkl_path = Path(file_path).name + "vocab_counts.pkl"

with open(pkl_path, "rb") as f:
    words_bag = pickle.load(f)

print(len(words_bag))

def train_bpe(vocab_size):
    """
    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab:
        The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes)
    merges:
        BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
        representing that <token1> was merged with <token2>.
        Merges are ordered by order of creation.
    """
    vocab, merges = {k: bytes([k]) for k in range(256)}, []
    words_bag_tuple = Counter({tuple(k): v for k, v in words_bag.items()})

    print("正在进行初始统计...")
    stats = Counter()
    for word, count in words_bag_tuple.items():
        for i in range(len(word) - 1):
            stats[(word[i], word[i + 1])] += count

    print("进入主循环...")
    for id in range(256, vocab_size):
        while True:
            if not stats:
                print("提前结束")
                return vocab, merges
            pair = stats.most_common(1)[0][0]
            if stats[pair] > 0:
                break
            else:
                del stats[pair]
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        vocab[id] = vocab[pair[0]] + vocab[pair[1]]

        new_words_bag_tuple = Counter()
        for word, count in words_bag_tuple.items():
            if pair[0] not in word:
                new_words_bag_tuple[word] = count
                continue
            new_tuple, k = [], 0
            while k < len(word):
                if k < len(word) - 1 and (word[k], word[k + 1]) == pair:
                    p0, p1 = pair
                    if k > 0:
                        stats[(word[k - 1], p0)] -= count
                        stats[(word[k - 1], id)] += count
                    if k + 2 < len(word):
                        stats[(p1, word[k + 2])] -= count
                        stats[(id, word[k + 2])] += count
                    stats[pair] -= count
                    new_tuple.append(id)
                    k += 2
                else:
                    new_tuple.append(word[k])
                    k += 1
            new_words_bag_tuple[tuple(new_tuple)] = count
        words_bag_tuple = new_words_bag_tuple

        if id % 100 == 0:
            print(f"\rMerge {id}/{vocab_size}", end="")
    return vocab, merges

output_pkl_path = Path(file_path).name + "tokenizer_model.pkl"

vocab, merges = train_bpe(32000)

model_data = {
    "vocab": vocab,
    "merges": merges
}

print(f"正在保存模型到 {output_pkl_path} ...")
with open(output_pkl_path, "wb") as f:
    pickle.dump(model_data, f)