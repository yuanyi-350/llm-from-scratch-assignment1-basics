import torch
import argparse
import os
from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer
from scripts.train import get_device

def load_custom_tokenizer(vocab_path, merges_path, special_tokens):
    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        raise FileNotFoundError(f"找不到词表文件: {vocab_path} 或 {merges_path}")
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens=special_tokens
    )
    return tokenizer


def load_model(ckpt_path, device):
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        device=device
    )

    print(f"Loading model from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device)

    state_dict = checkpoint['model_state_dict']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_p=0.9, device='cuda'):
    input_ids = tokenizer.encode(prompt)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.context_length:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
            next_token_id = idx_next.item()
            print(tokenizer.decode([next_token_id]), end="", flush=True)
            continue

        probs = softmax(logits, dim=-1)
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs = probs.masked_fill(indices_to_remove, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        next_token_id = idx_next.item()
        decoded_char = tokenizer.decode([next_token_id])
        print(decoded_char, end="", flush=True)

    print("\n" + "-" * 40)
    return idx.tolist()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help="Checkpoint path")
    parser.add_argument('--vocab_file', type=str, default='./data/tinystoriesv2-gpt4-train_vocab.pkl',
                        help="vocab path")
    parser.add_argument('--merges_file', type=str, default='./data/tinystoriesv2-gpt4-train_merges.pkl',
                        help="merges path")
    parser.add_argument('--prompt', type=str, default="Once upon a time", help="prompt")

    parser.add_argument('--temp', type=float, default=0.7, help="Temperature (0.0-1.0)")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p/Nucleus sampling probability (0.0-1.0)")
    parser.add_argument('--device', type=str, default='auto', help="Device")

    args = parser.parse_args()

    device = get_device(args.device)
    special_tokens = ["<|endoftext|>"]
    try:
        tokenizer = load_custom_tokenizer(args.vocab_file, args.merges_file, special_tokens)
    except Exception as e:
        print(f"Tokenizer 加载失败: {e}")
        return

    model = load_model(args.ckpt, device)
    generate(model, tokenizer, args.prompt, temperature=args.temp, device=device)


if __name__ == "__main__":
    main()