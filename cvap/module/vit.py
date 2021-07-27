import torch
from torch import nn
import torch.nn.functional as F

from clip import Transformer, LayerNorm 

class VisualTransformer(nn.Module):
    def __init__(
            self, 
            input_resolution: int, 
            patch_size: int, 
            width: int, 
            layers: int, 
            heads: int, 
            output_dim: int, 
            in_channels=3, 
            stride=None
        ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        stride = stride or patch_size
        if isinstance(stride, int):
            stride = [stride] * 2
        stride = list(stride)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=stride, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if isinstance(input_resolution, int):
            positions = (input_resolution // patch_size) ** 2 + 1
        else:
            row_stride, col_stride = stride[:2]
            nrow = (input_resolution[0] - patch_size) // row_stride + 1
            ncol = (input_resolution[1] - patch_size) // col_stride + 1
            positions = nrow * ncol + 1
            self.position_resolution = (nrow, ncol)
        self.positional_embedding = nn.Parameter(scale * torch.randn(positions, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x

