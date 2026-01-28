import pickle

with open('/home/yuanyi/data/cs336/Others/Stanford-CS336-spring25/assignment1-basics/tokenizer/tinystories_bpe_merges.pkl', 'rb') as f:
    data1 = pickle.load(f)

with open('/home/yuanyi/data/cs336/llm-from-scratch-assignment1-basics/data/tinystories_merges.pkl', 'rb') as f:
    data2 = pickle.load(f)

print(data1 == data2)

with open('/home/yuanyi/data/cs336/Others/Stanford-CS336-spring25/assignment1-basics/tokenizer/tinystories_bpe_vocab.pkl', 'rb') as f:
    data3 = pickle.load(f)

with open('/home/yuanyi/data/cs336/llm-from-scratch-assignment1-basics/data/tinystories_vocab.pkl', 'rb') as f:
    data4 = pickle.load(f)

print(data3 == data4)