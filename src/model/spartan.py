import math
from collections.abc import Callable
from functools import partial
from typing import Concatenate, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from timm.layers import trunc_normal_
from torchvision.ops import StochasticDepth


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_group: bool = True):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.use_group = use_group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_group:
            x = F.group_norm(
                x, num_groups=1, weight=self.weight, bias=self.bias, eps=self.eps
            )
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        norm: Optional[Callable[Concatenate[int, ...], nn.Module]] = nn.BatchNorm2d,
        act: Optional[Callable[Concatenate[torch.Tensor, ...], torch.Tensor]] = F.gelu,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )

        self.norm = norm(out_dim) if norm else nn.Identity()
        self.act = act if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SELayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        squeeze_dim: int,
        act: Callable[Concatenate[torch.Tensor, ...], torch.Tensor] = F.relu,
        scale_act: Callable[Concatenate[torch.Tensor, ...], torch.Tensor] = F.sigmoid,
    ):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Conv2d(in_dim, squeeze_dim, kernel_size=1)
        self.expand = nn.Conv2d(squeeze_dim, in_dim, kernel_size=1)
        self.act = act
        self.scale_act = scale_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.act(self.squeeze(scale))
        scale = self.scale_act(self.expand(scale))
        return x * scale


class SMixer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        conv: Callable[Concatenate[int, int, int, ...], ConvNormAct],
    ):
        assert in_dim % 2 == 0 and in_dim % 8 == 0
        super().__init__()

        int_dim = in_dim // 2

        self.gamma = nn.Parameter(1e-5 * torch.ones(1, int_dim, 1, 1))
        self.proj_in = nn.Conv2d(in_dim, int_dim, 1)
        self.h_conv = conv(int_dim, int_dim, 3, padding=1)
        self.l_conv = nn.Sequential(
            conv(
                int_dim,
                int_dim,
                3,
                stride=1,
                padding=2,
                dilation=2,
                norm=None,
                act=None,
            ),
            conv(int_dim, int_dim, 3, padding=2, dilation=2),
        )
        self.h_attn = SELayer(int_dim, int_dim // 4)
        self.l_attn = SELayer(int_dim, int_dim // 4)
        self.proj_out = nn.Conv2d(in_dim, in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(x)
        x = x + self.gamma * (x - F.adaptive_avg_pool2d(x, 1))
        x = torch.cat([self.h_attn(self.h_conv(x)), self.l_attn(self.l_conv(x))], dim=1)
        return self.proj_out(x)


class CMixer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        expand_ratio: int,
        conv: Callable[Concatenate[int, int, int, ...], ConvNormAct],
    ):
        assert expand_ratio > 1
        super().__init__()

        neck_dim = in_dim * expand_ratio
        use_group = conv.keywords["groups"] > 1

        self.theta = nn.Conv2d(in_dim, neck_dim, 1)
        self.proj = nn.Conv2d(in_dim, neck_dim, 1)
        self.complex = nn.Parameter(torch.randn(1, 1, 1, neck_dim // 2, 2))
        self.conv = conv(
            neck_dim,
            neck_dim,
            3,
            padding=1,
            groups=in_dim if use_group else neck_dim,
        )
        self.attn = SELayer(neck_dim, in_dim // 4)
        self.proj_out = nn.Conv2d(neck_dim, in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        x_ind = x.argmax(dim=1, keepdim=True)

        theta = F.gelu(self.theta(x))
        theta_max = theta.gather(dim=1, index=x_ind)
        weight = self.proj(x)
        weight_max = weight.gather(dim=1, index=x_ind)
        feat = weight * theta + weight_max * theta_max

        with torch.amp.autocast("cuda", enabled=False):
            complex_weight = self.complex.to(torch.float32)
            complex_weight = torch.view_as_complex(complex_weight)

            feat = feat.to(torch.float32)
            feat = feat.permute(0, 2, 3, 1).reshape(B, H, W, -1, 2).contiguous()
            feat = complex_weight * torch.view_as_complex(feat)
            feat = (
                torch.view_as_real(feat)
                .reshape(B, H, W, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

        x = self.conv(feat)
        x = self.attn(x)
        return self.proj_out(x)


class Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        smixer: Callable[Concatenate[int, ...], SMixer],
        cmixer: Callable[Concatenate[int, ...], CMixer],
        norm: Callable[Concatenate[int, ...], nn.Module] = nn.BatchNorm2d,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.smixer = smixer(in_dim)
        self.cmixer = cmixer(in_dim)

        self.dropout_1 = StochasticDepth(dropout, "row")
        self.dropout_2 = StochasticDepth(dropout, "row")

        self.norm_1 = norm(in_dim)
        self.norm_2 = norm(in_dim)

        self.scale_1 = nn.Parameter(1e-5 * torch.ones((1, in_dim, 1, 1)))
        self.scale_2 = nn.Parameter(1e-5 * torch.ones((1, in_dim, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout_1(self.scale_1 * self.smixer(self.norm_1(x)))
        x = x + self.dropout_2(self.scale_2 * self.cmixer(self.norm_2(x)))
        return x


class SpaRTAN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_layer: int,
        layer_depths: List[int],
        expand_ratios: List[int] = [],
        dims: List[int] = [],
        init_dim: int = 32,
        dropout: float = 0.0,
    ):
        assert num_layer == len(layer_depths)
        assert num_layer == len(dims) + 1
        super().__init__()

        conv_norm = partial(nn.BatchNorm2d, eps=1e-6)
        block_norm = partial(LayerNorm, eps=1e-6, use_group=True)
        act = F.gelu

        in_dim = init_dim
        block_idx = 0
        mid_point = num_layer // 2
        total_block = sum(layer_depths)
        layers = []
        for layer in range(num_layer):
            for _ in range(layer_depths[layer]):
                drop = dropout * block_idx / total_block
                block_idx += 1

                layers.append(
                    Block(
                        in_dim=in_dim,
                        smixer=partial(
                            SMixer,
                            conv=partial(
                                ConvNormAct,
                                norm=conv_norm,
                                act=act,
                                groups=1 if layer < mid_point else in_dim // 2,
                            ),
                        ),
                        cmixer=partial(
                            CMixer,
                            expand_ratio=expand_ratios[layer] if expand_ratios else 1,
                            conv=partial(
                                ConvNormAct,
                                norm=conv_norm,
                                act=act,
                                groups=1 if layer < mid_point else in_dim,
                            ),
                        ),
                        norm=block_norm,
                        dropout=drop,
                    )
                )

            if layer_depths[layer] != 0 and layer != num_layer - 1:
                layers.append(
                    ConvNormAct(
                        in_dim,
                        dims[layer] if dims else in_dim * 2,
                        2,
                        stride=2,
                        norm=conv_norm,
                        act=F.silu,
                    )
                )
                in_dim = dims[layer] if dims else in_dim * 2

        self.conv = nn.Sequential(
            ConvNormAct(
                3, init_dim // 2, 3, padding=1, stride=2, norm=conv_norm, act=act
            ),
            ConvNormAct(
                init_dim // 2, init_dim, 3, padding=1, stride=2, norm=conv_norm
            ),
        )

        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(in_dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(start_dim=1)
        return self.linear(x)


def build(num_classes: int, config: DictConfig) -> nn.Module:
    return SpaRTAN(num_classes, **config)
