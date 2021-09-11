from fvcore.common.registry import Registry
from collections import OrderedDict
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from clip import Bottleneck 
from clip import QuickGELU, LayerNorm

ENCODER_MODULES_REGISTRY = Registry("ENCODER_MODULES")
ENCODER_MODULES_REGISTRY.__doc__ = """
Registry for encoder modules.
"""

def build_encoder_module(cfg, **kwargs):
    return ENCODER_MODULES_REGISTRY.get(cfg.name)(cfg, **kwargs)

class MetaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.position_resolution = None

    @property
    def hp(self): 
        return [] 

    @hp.setter
    def hp(self, hp):
        pass 

class Miscellanea(MetaEncoder):
    """ a parameter container.  
    """
    def __init__(self, cfg, position_resolution=None, **kwargs):
        super().__init__()
        if position_resolution is not None:
            width = position_resolution[-1]
            self.position_resolution = position_resolution[:-1]
            positions = np.prod(self.position_resolution) + 1
        else:
            self.position_resolution = None 
            width, positions = 0, 0 
        scale = width ** -0.5 if width > 0 else 0
        self.positional_embedding = nn.Parameter(scale * torch.randn(positions, width))
        self.class_embedding = nn.Parameter(scale * torch.randn(width)) #None # `<s>` as the class 

    def initialize_parameters(self):
        pass

@ENCODER_MODULES_REGISTRY.register()
class AddonEncoder(nn.Module):
    """ enhance an existing encoder.
    """
    def __init__(self, cfg, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, **kwargs):
        return x

@ENCODER_MODULES_REGISTRY.register()
class CLIPMisc(Miscellanea):
    """ a parameter container.  
    """
    def __init__(self, cfg, position_resolution=None, **kwargs):
        super().__init__(cfg, position_resolution=position_resolution, **kwargs)
        pass

    def replace_modules(self, reference, keep_hp=False):
        self.positional_embedding, self.class_embedding = reference.positional_embedding, reference.class_embedding
        if not keep_hp:
            self.position_resolution = reference.position_resolution

    @property
    def hp(self):
        return [self.position_resolution]

    @hp.setter
    def hp(self, hp):
        (self.position_resolution,) = hp

    @property
    def pos_embedding(self):
        positional_embedding = interp_clip_vp_embedding(self.positional_embedding, self.position_resolution)
        #print(f"{self.positional_embedding.shape} {self.position_resolution} {positional_embedding.shape}")
        return positional_embedding

    @property
    def cls_embedding(self):
        return self.class_embedding 

@ENCODER_MODULES_REGISTRY.register()
class GPTPreEncoder(MetaEncoder):
    def __init__(self, cfg, width=512, ctx_len=77, **kwargs):
        super().__init__()
        self.position_resolution = (ctx_len, width) 
        self.token_embedding = nn.Embedding(cfg.vocab_size, width)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    def forward(
        self, 
        x: torch.Tensor, 
        positional_embedding: torch.Tensor = None,
        class_embedding: torch.Tensor = None, **kwargs
    ):
        x = self.token_embedding(x).type(self.dtype)  # [batch_size, n_ctx, d_model]
        positional_embedding = positional_embedding[:x.shape[1]]
        x = x + positional_embedding.type(self.dtype)
        return x

@ENCODER_MODULES_REGISTRY.register()
class GPTPostEncoder(MetaEncoder):
    def __init__(self, cfg, width=512, embed_dim=512, **kwargs):
        super().__init__()
        scale = width ** -0.5
        self.ln = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        positional_embedding: torch.Tensor = None,
        class_embedding: torch.Tensor = None, **kwargs
    ):
        x = self.ln(x[:, 0, :])
        x = x @ self.proj
        return x

def _vit_position_resolution(input_resolution, patch_size, stride):
    stride = stride or patch_size
    if isinstance(stride, int):
        stride = [stride] * 2
    stride = list(stride)

    if isinstance(input_resolution, int):
        nrow = ncol = input_resolution // patch_size
        positions = nrow ** 2 + 1 # 
        position_resolution = (nrow, ncol)
    else:
        row_stride, col_stride = stride[:2]
        nrow = (input_resolution[0] - patch_size) // row_stride + 1
        ncol = (input_resolution[1] - patch_size) // col_stride + 1
        positions = nrow * ncol + 1
        position_resolution = (nrow, ncol)
    return stride, positions, position_resolution

def interp_conv_weight_channel(conv_weight, input_shape):
    if conv_weight.shape[1] != input_shape[1]:
        input_shape = (conv_weight.shape[0], input_shape[1])
        conv_weight = conv_weight.permute(2, 3, 0, 1)
        conv_weight = F.interpolate(
            conv_weight,
            input_shape,
            mode="bilinear",
            align_corners=False,
        )
        conv_weight = conv_weight.permute(2, 3, 0, 1)
    return conv_weight

def interp_conv_weight_spatial(conv_weight, patch_shape):
    if conv_weight.shape[-2:] != patch_shape:
        conv_weight = F.interpolate(
            conv_weight,
            patch_shape,
            mode="bilinear",
            align_corners=False,
        )
    return conv_weight

@ENCODER_MODULES_REGISTRY.register()
class ViTPreEncoder(MetaEncoder):
    def __init__(self, cfg, width=768, resolution=224, **kwargs): 
        super().__init__()
        self.stride, _, self.position_resolution = _vit_position_resolution(
            resolution, cfg.patch_size, cfg.stride
        )
        self.position_resolution += (width,)
        self.conv1 = nn.Conv2d(
            in_channels=cfg.in_channels, out_channels=width, kernel_size=cfg.patch_size, stride=self.stride, bias=False
        )
        self.patch_size = self.conv1.weight.shape[-2:]
        self.ln = LayerNorm(width)
        self.initialize_parameters()

    def initialize_parameters(self):
        pass

    def replace_modules(self, reference, keep_hp=False):
        self.conv1, self.ln = reference.conv1, reference.ln
        if not keep_hp:
            self.stride, self.patch_size, self.position_resolution = \
                reference.stride, reference.patch_size, reference.position_resolution

    @property
    def hp(self):
        return [self.stride, self.patch_size, self.position_resolution]

    @hp.setter
    def hp(self, hp):
        (self.stride, self.patch_size, self.position_resolution) = hp

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(
        self, 
        x: torch.Tensor, 
        positional_embedding: torch.Tensor = None,
        class_embedding: torch.Tensor = None, **kwargs
    ):
        assert x.dim() == 4, f"expect 4d `x` but get x.dim == {x.dim()}"
        x = x.type(self.dtype)
        if x.shape[1] != 3: # interpolate weight
            use_mean = True
            conv1_weight = interp_conv_weight_spatial(self.conv1.weight, self.patch_size)
            #print(f"{self.conv1.weight.shape}, {conv1_weight.shape}, {self.patch_size}, {self.conv1.stride}, {self.stride}")
            conv1_weight = (
                conv1_weight.mean(1, keepdim=True) if use_mean else
                interp_conv_weight_channel(conv1_weight, x.shape)
            )
            x = F.conv2d(
                x, conv1_weight, bias=self.conv1.bias, stride=self.stride
            )
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            #print(f"{self.conv1.weight.shape}, {self.patch_size}, {self.conv1.stride}, {self.stride}")
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([
            class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        #print(f"C {x.shape}, {positional_embedding.shape}, {self.position_resolution}")
        x = x + positional_embedding[:x.shape[1]].to(x.dtype)
        x = self.ln(x)
        return x

@ENCODER_MODULES_REGISTRY.register()
class ViTPostEncoder(MetaEncoder):
    def __init__(self, cfg, width=768, embed_dim=512, **kwargs):
        super().__init__()
        scale = width ** -0.5
        self.ln = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, embed_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        pass
    
    def forward(
        self, 
        x: torch.Tensor, 
        positional_embedding: torch.Tensor = None,
        class_embedding: torch.Tensor = None, **kwargs
    ):
        x = self.ln(x[:, 0, :])
        x = x @ self.proj
        return x

def _resnet_position_resolution(input_resolution, patch_size=32, stride=None):
    stride = stride or patch_size
    if isinstance(stride, int):
        stride = [stride] * 2
    stride = list(stride)

    if isinstance(input_resolution, int):
        nrow = ncol = input_resolution // patch_size
        positions = nrow ** 2 + 1 # 
        position_resolution = (nrow, ncol)
    else:
        row_stride, col_stride = stride[:2]
        nrow = (input_resolution[0] - 0) // row_stride
        ncol = (input_resolution[1] - 0) // col_stride
        positions = nrow * ncol + 1
        position_resolution = (nrow, ncol)
    return stride, positions, position_resolution

@ENCODER_MODULES_REGISTRY.register()
class ResNetPreEncoder(MetaEncoder):
    def __init__(self, cfg, width=64, **kwargs):
        super().__init__()
        # the 3-layer stem
        self.conv1 = nn.Conv2d(cfg.in_channels, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.initialize_parameters()

    def initialize_parameters(self):
        pass

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(
        self, 
        x: torch.Tensor, 
        positional_embedding: torch.Tensor = None,
        class_embedding: torch.Tensor = None, **kwargs
    ):
        assert x.dim() == 4, f"expect 4d `x` but get x.dim == {x.dim()}"
        x = x.type(self.dtype)
        if x.shape[1] != 3: # interpolate weight
            use_mean = True
            conv1_weight = (
                self.conv1.weight.mean(1, keepdim=True) if use_mean else
                interp_conv_weight_channel(self.conv1.weight, x.shape)
            )
            x = F.conv2d(
                x, conv1_weight, bias=self.conv1.bias, stride=self.conv1.stride, padding=self.conv1.padding
            )
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = self.relu(self.bn1(x))
        for conv, bn in [(self.conv2, self.bn2), (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x

@ENCODER_MODULES_REGISTRY.register()
class ResNetPostEncoder(MetaEncoder):
    def __init__(self, cfg, width=64, embed_dim=None, resolution=224, **kwargs):
        super().__init__()
        width = width * 32  # the ResNet feature dimension
        self.num_heads = width // 64

        _, _, self.position_resolution = _resnet_position_resolution(resolution)
        self.position_resolution += (width,)

        self.k_proj = nn.Linear(width, width)
        self.q_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.c_proj = nn.Linear(width, embed_dim or width)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.q_proj.weight, std=std)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)

    def replace_modules(self, reference, keep_hp=False):
        self.k_proj, self.q_proj, self.v_proj, self.c_proj = (
            reference.k_proj, reference.q_proj, reference.v_proj, reference.c_proj
        )
        if not keep_hp:
            self.position_resolution = reference.position_resolution

    @property
    def hp(self):
        return [self.position_resolution] 

    @hp.setter
    def hp(self, hp):
        (self.position_resolution,) = hp 

    def forward(
        self, 
        x: torch.Tensor, 
        positional_embedding: torch.Tensor = None,
        class_embedding: torch.Tensor = None, **kwargs
    ):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        #print(f"C {x.shape}, {positional_embedding.shape}, {self.position_resolution}")
        x = x + positional_embedding[:x.shape[0], None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]

@ENCODER_MODULES_REGISTRY.register()
class ResNetBackbone(MetaEncoder):
    def __init__(self, cfg, width=64, **kwargs):
        super().__init__()
        self.batch_first = True 
        layers = cfg.layers

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.initialize_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def initialize_parameters(self):
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

@ENCODER_MODULES_REGISTRY.register()
class TransformerBackbone(MetaEncoder):
    def __init__(self, cfg, width=512, ctx_len=77, **kwargs):
        super().__init__()
        self.batch_first = False
        self.ctx_len = ctx_len
        heads = width // 64
        
        attn_mask = self.build_attention_mask()

        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width, heads, attn_mask, cfg.skip_attn_mask
            ) for _ in range(cfg.layers)
        ])

    def build_attention_mask(self):
        if self.ctx_len is None:
            return None
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.ctx_len, self.ctx_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, x: torch.Tensor, **kwargs):
        return self.resblocks(x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, skip_attn_mask: bool = True):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.skip_attn_mask = skip_attn_mask

    def attention(self, x: torch.Tensor):
        if not self.skip_attn_mask and self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)    
            attn_mask = self.attn_mask[:x.shape[0], :x.shape[0]]
        else:
            attn_mask = None 
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def interp_clip_vp_embedding(old_pos_emb, pos_resolution, bop=1):
    """ vp: stands for `visual positional`
        bop: start position of the postional embeddings
        old_pos_emb: (H x W + 1, D)
    """
    num_pos, pos_dim = old_pos_emb.shape[-2:]
    num_pos_required = np.prod(pos_resolution)
    # TODO assumed old_pos_emb comes from vision pos, but it can come from audio pos
    # if these two kinds do not share, we do not need to interp the input pos.
    # FIXME adhoc: the condition of not sharing may be wrong.
    if num_pos_required + 1 == num_pos:
        return old_pos_emb
    # old_pos_emb must be vision pos if sharing pos between vision and audio
    num_pos = int(np.sqrt(num_pos - bop))
    ptensor = old_pos_emb[bop:].reshape(
        -1, num_pos, num_pos, pos_dim
    ).permute(0, 3, 1, 2)
    if ptensor.shape[-2:] != pos_resolution:
        new_pos_emb = F.interpolate(
            ptensor,
            pos_resolution,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_emb = torch.cat((
            old_pos_emb[:bop], new_pos_emb.view(-1, pos_dim)
        ), dim=0)
    else: # do nothing
        new_pos_emb = old_pos_emb
    return new_pos_emb
