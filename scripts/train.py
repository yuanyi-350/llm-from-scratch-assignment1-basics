#!/usr/bin/env python3
"""
Training script for Transformer language model with wandb and tqdm monitoring.

Example usage:
    # Train with wandb logging
    python train.py --data_dir ./data --wandb_project "my-project" --wandb_run_name "experiment-1"

    # Train without wandb
    python train.py --data_dir ./data --no_wandb
"""
import argparse
from tqdm import tqdm
import wandb
from contextlib import nullcontext
from cs336_basics.model import *


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Transformer language model')

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=10000, help='Size of vocabulary')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--d_ff', type=int, default=1344, help='FFN dimension')
    parser.add_argument('--context_len', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')

    # Optimizer arguments
    parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate')
    parser.add_argument('--warm_up_it', type=int, default=500, help='Warmup iterations')
    parser.add_argument('--cosine_it', type=int, default=10000, help='Cosine annealing iterations')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='Adam beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_steps', type=int, default=6000, help='Total training steps')
    parser.add_argument('--val_interval', type=int, default=100, help='Validation interval')
    parser.add_argument('--val_batches', type=int, default=10, help='Number of validation batches')
    parser.add_argument('--save_intervals', type=int, default=1000, help='Checkpoint save interval')
    parser.add_argument('--log_intervals', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_ckp_path', type=str, default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--resume_ckp', type=str, default=None, help='Path to checkpoint to resume from')

    # Data arguments
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory path')
    parser.add_argument('--precision', type=str, default='fp32', choices=['bf16', 'fp32'],
                        help='Mixed precision mode')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto, cpu, cuda, mps')

    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='cs336-transformer', help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')

    # Ablation Arguments
    parser.add_argument('--norm_type', type=str, default='pre', choices=['pre', 'post', 'none'],
                        help='Type of layer normalization: pre, post, or none')
    parser.add_argument('--disable_rope', action='store_true',
                        help='Disable Rotary Positional Embeddings (NoPE)')
    parser.add_argument('--activation_type', type=str, default='swiglu', choices=['swiglu', 'silu'],
                        help='Activation function type: swiglu or silu')

    return parser.parse_args()


def get_device(device_arg):
    """Get the appropriate device"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    return device_arg


def get_dataset_memmap(path, dtype=np.uint16):
    """Load dataset using memory mapping for efficiency"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = np.memmap(path, dtype=dtype, mode='r')
    return dataset


def main():
    args = parse_args()

    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")

    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
        print(f"Wandb initialized: {wandb.run.name}")

    # Create checkpoint directory
    os.makedirs(args.save_ckp_path, exist_ok=True)

    actual_d_ff = args.d_ff
    if args.activation_type == 'silu' and args.d_ff == 1344:  # 假设用户没改 d_ff
        actual_d_ff = 4 * args.d_model
        print(f"Switching d_ff to {actual_d_ff} for SiLU to match parameter count.")

    # Initialize model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=actual_d_ff,
        norm_type=args.norm_type,
        activation_type=args.activation_type,
        use_rope=not args.disable_rope,
        device=device
    )

    use_amp = (device == "cuda" and args.precision == "bf16")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params} parameters")

    # Log model info to wandb
    if not args.no_wandb:
        wandb.log({'model/total_parameters': total_params})

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )

    # Load datasets
    if args.data_dir is None:
        raise ValueError("Data directory must be specified with --data_dir")

    train_data_path = os.path.join(args.data_dir, 'train.dat')
    val_data_path = os.path.join(args.data_dir, 'valid.dat')

    train_data = get_dataset_memmap(train_data_path)
    val_data = get_dataset_memmap(val_data_path)

    print(f"Train data size: {len(train_data)} tokens")
    print(f"Val data size: {len(val_data)} tokens")

    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_ckp:
        print(f"Resuming from checkpoint: {args.resume_ckp}")
        start_iter = load_checkpoint(args.resume_ckp, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    model.train()
    train_losses = []

    # Create progress bar
    pbar = tqdm(range(start_iter, args.train_steps),
                desc="Training",
                initial=start_iter,
                total=args.train_steps)

    for iter_num in pbar:
        # Get learning rate for this iteration
        lr = get_lr_cosine_schedule(
            iter_num,
            args.max_lr,
            args.min_lr,
            args.warm_up_it,
            args.cosine_it
        )

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Sample batch
        input_ids, target_ids = get_batch(
            train_data,
            args.batch_size,
            args.context_len,
            device=device
        )

        input_ids = input_ids.long()
        target_ids = target_ids.long()

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(input_ids)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1))

        loss.backward()
        gradient_clipping(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        # Track loss
        train_losses.append(loss.item())

        # Logging
        if iter_num % args.log_intervals == 0:
            avg_loss = np.mean(train_losses[-100:]) if len(train_losses) >= 100 else np.mean(train_losses)
            perplexity = np.exp(avg_loss)

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{avg_loss:.4f}',
                'PPL': f'{perplexity:.2f}',
                'LR': f'{lr:.2e}'
            })

            # Log to wandb
            if not args.no_wandb:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/avg_loss': avg_loss,
                    'train/perplexity': perplexity,
                    'train/learning_rate': lr,
                    'iteration': iter_num
                })

        # Validation
        if iter_num % args.val_interval == 0 and iter_num > 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for _ in range(args.val_batches):
                    val_input_ids, val_target_ids = get_batch(
                        val_data,
                        args.batch_size,
                        args.context_len,
                        device=device
                    )
                    with amp_ctx:
                        val_logits = model(val_input_ids)
                        val_logits_flat = val_logits.view(-1, val_logits.size(-1))
                        val_targets_flat = val_target_ids.view(-1)
                        val_loss = cross_entropy(val_logits_flat, val_targets_flat)
                    val_losses.append(val_loss.item()) # 只保存数值
                    # 坑点: val_losses.append(val_loss) 保存了整个计算图！

            avg_val_loss = np.mean(val_losses)
            val_perplexity = np.exp(avg_val_loss)

            # Log validation metrics
            tqdm.write(f"Validation | Loss: {avg_val_loss:.4f} | PPL: {val_perplexity:.2f}")

            # Log to wandb
            if not args.no_wandb:
                wandb.log({
                    'val/loss': avg_val_loss,
                    'val/perplexity': val_perplexity,
                    'iteration': iter_num
                })

            model.train()

        # Save checkpoint
        if iter_num % args.save_intervals == 0 and iter_num > 0:
            checkpoint_path = os.path.join(args.save_ckp_path, f'checkpoint_{iter_num}.pt')
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

    # Close progress bar
    pbar.close()

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.save_ckp_path, f'checkpoint_final_{args.train_steps}.pt')
    save_checkpoint(model, optimizer, args.train_steps, final_checkpoint_path)
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    print("Training completed!")

    # Finish wandb
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
