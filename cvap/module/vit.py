import torch
from torch import nn
import torch.nn.functional as F

from clip import LayerNorm
from . import Transformer

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
        if isinstance(patch_size, int):
            patch_size = [patch_size] * 2
        stride = list(stride)
        patch_size = list(patch_size)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=stride, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if isinstance(input_resolution, int):
            positions = (input_resolution // patch_size[0]) ** 2 + 1
        else:
            row_stride, col_stride = stride[:2]
            nrow = (input_resolution[0] - patch_size[0]) // row_stride + 1
            ncol = (input_resolution[1] - patch_size[1]) // col_stride + 1
            positions = nrow * ncol + 1
            self.position_resolution = (nrow, ncol)
        self.positional_embedding = nn.Parameter(scale * torch.randn(positions, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor, require_feature: bool=False):
        x = x.type(self.dtype)
        if x.shape[1] != self.conv1.weight.shape[1]: # interpolate weight
            conv1_weight = self.conv1.weight.mean(1, keepdim=True)
            x = F.conv2d(
                x, conv1_weight, bias=self.conv1.bias, stride=self.conv1.stride
            )
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x_feature = self.ln_post(x)

        if self.proj is not None:
            x = x[:, 0, :] @ self.proj

        if require_feature:
            return x, x_feature[:, 1:]

        return x

