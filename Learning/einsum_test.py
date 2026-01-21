import torch
from einops import rearrange, einsum

device = torch.device("xpu")
print(f"正在使用的设备: {device}")

images = torch.randn(64, 128, 128, 3, device=device)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10, device=device)

dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")

images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

dimmed_images_einsum = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)

print(f"结果 Tensor 的设备: {dimmed_images_einsum.device}")