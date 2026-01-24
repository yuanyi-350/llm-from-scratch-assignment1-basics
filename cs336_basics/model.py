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

        exponent = torch.arange(0, d_k, 2, dtype=torch.float, device=device) / d_k
        freqs = 1.0 / (theta ** exponent)  # Shape: (d_k / 2,)
        idx_theta = torch.arange(max_seq_len, device=device, dtype=torch.float)
        angles = torch.outer(idx_theta, freqs)

        cos_cached = torch.cos(angles).unsqueeze(-1)
        sin_cached = torch.sin(angles).unsqueeze(-1)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor and return a tensor of the same shape.

        Args:
            x: shape (..., seq_len, d_k)
            token_positions: shape (..., seq_len) specifying the token positions.
        """
        batch_seq_shape = x.shape[:-1]
        x_reshaped = x.view(*batch_seq_shape, self.d_k // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 去掉最后一个维度以便计算 (..., seq_len, d_k/2)
        cos = cos.squeeze(-1)
        sin = sin.squeeze(-1)
        x1_new = x1 * cos - x2 * sin
        x2_new = x1 * sin + x2 * cos
        x_out = torch.stack((x1_new, x2_new), dim=-1)

        x_out = x_out.flatten(-2)

        return x_out