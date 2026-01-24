import torch
import torch.nn as nn
import math
from einops import einsum, rearrange


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
    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        hidden_dim = int(8 * d_model / 3)
        self.d_ff = ((hidden_dim + 63) // 64) * 64
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
        batch_seq_shape = x.shape[:-1] # (..., seq_len)
        x_reshaped = x.view(*batch_seq_shape, self.d_k // 2, 2)
        x1 = x_reshaped[..., 0] # (..., seq_len, d_k/2)
        x2 = x_reshaped[..., 1] # (..., seq_len, d_k/2)
        cos = self.cos_cached[token_positions] # shape: (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions] # shape: (..., seq_len, d_k/2)

        x1_new = x1 * cos - x2 * sin # (..., seq_len, d_k/2)
        x2_new = x1 * sin + x2 * cos # (..., seq_len, d_k/2)
        x_out = torch.stack((x1_new, x2_new), dim=-1) # (..., seq_len, d_k/2, 2)
        return x_out.flatten(-2) # (..., seq_len, d_k)



def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: torch.Tensor Input of the softmax
    dim: int The dimension of x that you want to impelement softmax to.
    """
    x = x - torch.max(x, dim=dim, keepdim=True).values
    x = torch.exp(x)
    return x / torch.sum(x, dim=dim, keepdim=True)



def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                 mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    scores = softmax(scores, dim=-1)
    return einsum(scores, V, '... s_q s_k, ... s_k d -> ... s_q d')



class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: The dimension of the input vectors (e.g., 4096).
            num_heads: The number of attention heads (e.g., 32).
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # d_k = d_v = d_model / h

        # 1. 定义投影矩阵 W_Q, W_K, W_V
        # 题目要求总共三次矩阵乘法
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 2. 定义输出投影 W_O
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
            self,
            x: torch.Tensor,
            rope_module=None,  # 假设 RoPE 模块作为参数传入
            token_positions: torch.Tensor = None  # 用于 RoPE 的位置索引
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            rope_module: 你的 RotaryPositionalEmbedding 实例
            token_positions: 用于计算 RoPE 频率的位置 ID
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影 (Project)
        # Shape: (B, S, d_model) -> (B, S, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. 切分多头 (Split Heads)
        # 将 d_model 拆分为 (num_heads, d_head)
        # Shape: (B, S, H, D)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)

        # 3. 应用 RoPE (Apply RoPE)
        # 题目要求：Apply to Query and Key, but NOT Value
        # 题目要求：Head dimension should be handled as batch dimension
        if rope_module is not None and token_positions is not None:
            # 需要调整形状以适配你之前的 RoPE 实现: (..., seq_len, d_head)
            # 这里我们要把 heads 放到 batch 维度或者作为额外的独立维度处理
            q_in = rearrange(q, 'b s h d -> b h s d')
            k_in = rearrange(k, 'b s h d -> b h s d')

            # 应用 RoPE (假设 rope_module.forward 返回变换后的值)
            # 注意：这里调用的是你之前实现的 forward
            q_rotated = rope_module(q_in, token_positions)
            k_rotated = rope_module(k_in, token_positions)

            # 转回 (B, S, H, D) 以便做 Attention
            q = rearrange(q_rotated, 'b h s d -> b s h d')
            k = rearrange(k_rotated, 'b h s d -> b s h d')

        # 4. 构建因果掩码 (Causal Masking)
        # 题目要求：Mask value of True means attend (keep), False means ignore
        # 我们需要一个下三角矩阵 (Lower Triangular)
        # shape: (seq_len, seq_len)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        # 5. 计算 Scaled Dot-Product Attention
        # 此时 q, k, v 都是 (B, S, H, D)，我们需要把 H 移到 Batch 维度或者使用 einsum 处理
        # 既然我们之前的 SDPA 实现支持广播，我们可以调整维度顺序为 (B, H, S, D)
        # 这是一个常见的做法，或者直接使用 einops

        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        # 调用你之前写好的函数
        # Mask 会自动广播到 (B, H, S, S)
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        # attn_out shape: (B, H, S, D)

        # 6. 拼接多头 (Merge Heads)
        # Shape: (B, H, S, D) -> (B, S, H*D) = (B, S, d_model)
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')

        # 7. 输出投影 (Output Projection)
        output = self.w_o(attn_out)

        return output