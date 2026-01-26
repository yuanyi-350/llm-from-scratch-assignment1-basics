import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
import math
from einops import einsum, rearrange
from typing import Union, BinaryIO, IO


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features), device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm.to(dtype=input_dtype) * self.weight



class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, device=None, dtype=None):
        super().__init__()
        hidden_dim = int(8 * d_model / 3)
        self.d_ff = d_ff if d_ff else ((hidden_dim + 63) // 64) * 64
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        gate = x1 * torch.sigmoid(x1)
        hidden = gate * self.w3(x)
        return self.w2(hidden)



class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: float, Theta value for the RoPE (base frequency).
            d_k: int, dimension of query and key vectors.
            max_seq_len: int, Maximum sequence length that will be inputted.
            device: torch.device | None, Device to store the buffer on.
        """
        super().__init__()
        self.d_k = d_k
        self.theta = theta

        exponent = torch.arange(0, d_k, 2, device=device, dtype=torch.float) / d_k
        freqs = 1.0 / (theta ** exponent)  # shape: (d_k / 2,)
        idx = torch.arange(max_seq_len, device=device, dtype=torch.float) # shape: (max_seq_len,)
        angles = torch.outer(idx, freqs) # angles[i][k] = i / (Theta ^ ((2k-2)/d))

        cos_cached = torch.cos(angles) # shape: (max_seq_len, d_k/2)
        sin_cached = torch.sin(angles) # shape: (max_seq_len, d_k/2)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.

        Args:
            x: shape (..., seq_len, d_k)
            token_positions: shape (..., seq_len) specifying the token positions.
        """
        x_reshaped = rearrange(x, '... (d c) -> ... d c', c=2)
        x1 = x_reshaped[..., 0] # shape: (..., d/2)
        x2 = x_reshaped[..., 1] # shape: (..., d/2)

        cos = self.cos_cached[token_positions] # shape: (..., d/2)
        sin = self.sin_cached[token_positions] # shape: (..., d/2)

        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        x_out = torch.stack((x1_new, x2_new), dim=-1)
        return rearrange(x_out, '... d c -> ... (d c)')



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: torch.Tensor Input of the softmax
    dim: int The dimension of x that you want to impelement softmax to.
    """
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)



def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Given key (k), query (q), and value (v) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        q (Float[Tensor, " ... queries d_k"]): Query tensor
        k (Float[Tensor, " ... keys d_k"]): Key tensor
        v (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = q.shape[-1]
    scores = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    scores = softmax(scores, dim=-1)
    return einsum(scores, v, '... s_q s_k, ... s_k d -> ... s_q d')


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        Args:
            d_model: The dimension of the input vectors (e.g., 4096).
            num_heads: The number of attention heads (e.g., 32).
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope_module=None, token_positions: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=self.num_heads)

        if rope_module is not None and token_positions is not None:
            q = rope_module(q, token_positions)
            k = rope_module(k, token_positions)

        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        return self.w_o(attn_out)



class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = CausalMultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)

        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, rope_module=None, token_positions=None):
        h = x
        x_norm = self.norm1(x)
        attn_out = self.mha(x_norm, rope_module=rope_module, token_positions=token_positions)
        h = h + attn_out
        ffn_out = self.ffn(self.norm2(h))
        return h + ffn_out



def cross_entropy(logits: torch.Tensor, true_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross entropy loss efficiently and numerically stably.
    Args:
        logits (torch.Tensor): The raw output of the model (before softmax).
                               Shape: (batch_size, ..., vocab_size)
        true_labels (torch.Tensor): The ground truth indices.
                                    Shape: (batch_size, ...)
    Returns:
        torch.Tensor: The average loss across the batch.
    """
    c = logits.max(dim=-1, keepdim=True).values
    logits_stable = logits - c
    exp_sum = torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True)
    log_sum_exp = c + torch.log(exp_sum)

    true_logits = logits.gather(dim=-1, index=true_labels.unsqueeze(-1))
    loss = log_sum_exp - true_logits
    return loss.mean()




class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']  # alpha in algorithm
            beta1, beta2 = group['betas']
            eps = group['eps']  # epsilon
            weight_decay = group['weight_decay']  # lambda

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                state = self.state[p]

                # State initialization (m_0 = 0, v_0 = 0)
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)  # m
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # v

                m = state['exp_avg']
                v = state['exp_avg_sq']

                state['step'] += 1
                t = state['step']

                # m <- beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # v <- beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # alpha_t calculation
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                # alpha_t = alpha * sqrt(1 - beta2^t) / (1 - beta1^t)
                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                # theta <- theta - alpha_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                # theta <- theta - alpha * lambda * theta
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
        return loss



def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, T_w: int, T_c: int) -> float:
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t <= T_c:
        progress = (t - T_w) / (T_c - T_w)
        cosine_decay = 0.5 * (1 + math.cos(progress * math.pi))
        return alpha_min + cosine_decay * (alpha_max - alpha_min)
    else:
        return alpha_min



def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm_sq = sum(p.grad.data.norm(2).item() ** 2 for p in params_with_grad)
    total_norm = total_norm_sq ** 0.5
    if total_norm >= max_norm:
        scale_factor = max_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(scale_factor)


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = len(x)
    high = data_len - context_length

    ix = torch.randint(low=0, high=high, size=(batch_size,))
    x_batch_np = [x[i: i + context_length] for i in ix]
    y_batch_np = [x[i + 1: i + context_length + 1] for i in ix]

    x_batch = torch.tensor(np.array(x_batch_np), dtype=torch.long, device=device)
    y_batch = torch.tensor(np.array(y_batch_np), dtype=torch.long, device=device)
    return x_batch, y_batch

