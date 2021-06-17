import torch
from torch import nn

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
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=stride, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if isinstance(input_resolution, int):
            positions = (input_resolution // patch_size) ** 2 + 1
        else:
            nrow = (input_resolution[0] - patch_size) // stride + 1
            ncol = (input_resolution[1] - patch_size) // stride + 1
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

class LARS(torch.optim.Optimizer):
    def __init__(
            self, params, lr, 
            weight_decay=0, 
            momentum=0.9, 
            eta=0.001,
            weight_decay_filter=None, 
            lars_adaptation_filter=None
        ):
        defaults = dict(
            lr=lr, 
            weight_decay=weight_decay, 
            momentum=momentum,
            eta=eta, 
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad
                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0., torch.where(
                            update_norm > 0,
                            (g['eta'] * param_norm / update_norm), one
                        ), one
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

