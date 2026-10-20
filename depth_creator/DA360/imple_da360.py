# Inference-only implementation of DA360 (unofficial).
#
# Derived from:
#   - DA360              MIT         (c) 2025 Insta360 Research Team
#   - Depth Anything V2  Apache-2.0
#   - DINOv2             Apache-2.0  (c) Meta Platforms, Inc. and affiliates
#   - MiDaS              MIT         (c) 2019 Intel ISL
#
# Modified for inference. Full license texts: THIRD_PARTY_LICENSES.txt
#
# NOTICE: This file contains only the inference part of the DA360 model.
# Training code has been removed, so it is not suitable for training or fine-tuning.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

PATCH_SIZE = 14
PRETRAIN_GRID = 37

ENCODER_CONFIGS = {
    "vits": {"embed_dim": 384, "depth": 12, "num_heads": 6, "features": 64,
             "out_channels": [48, 96, 192, 384], "layer_ids": [2, 5, 8, 11]},
    "vitb": {"embed_dim": 768, "depth": 12, "num_heads": 12, "features": 128,
             "out_channels": [96, 192, 384, 768], "layer_ids": [2, 5, 8, 11]},
    "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "features": 256,
             "out_channels": [256, 512, 1024, 1024], "layer_ids": [4, 11, 17, 23]},
}


# --------------------------------------------------------------------------- #
# DINOv2 backbone
# --------------------------------------------------------------------------- #
class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        assert H % PATCH_SIZE == 0 and W % PATCH_SIZE == 0, f"input {H}x{W} must be a multiple of {PATCH_SIZE}"
        return self.proj(x).flatten(2).transpose(1, 2)


class DinoVisionTransformer(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, layer_ids):
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_ids = layer_ids

        self.patch_embed = PatchEmbed(embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, PRETRAIN_GRID**2 + 1, embed_dim))
        self.blocks = nn.ModuleList(Block(embed_dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x: Tensor, h: int, w: int) -> Tensor:
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if num_patches == N and h == w:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed, patch_pos_embed = pos_embed[:, :1], pos_embed[:, 1:]
        dim = x.shape[-1]
        h0, w0 = h // PATCH_SIZE + 0.1, w // PATCH_SIZE + 0.1
        sqrt_N = math.sqrt(N)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / sqrt_N, w0 / sqrt_N),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1).to(x.dtype)

    def get_intermediate_layers(self, x: Tensor):
        h, w = x.shape[-2:]
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, h, w)

        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.layer_ids:
                outputs.append(x)
        outputs = [self.norm(out) for out in outputs]
        return [(out[:, 1:], out[:, 0]) for out in outputs]


# --------------------------------------------------------------------------- #
# DPT head (equirectangular padding)
# --------------------------------------------------------------------------- #
def erp_pad(x: Tensor, pad: int) -> Tensor:
    w = x.shape[-1]
    top = torch.flip(torch.roll(x[:, :, :pad, :], w // 2, -1), dims=[-2])
    bottom = torch.flip(torch.roll(x[:, :, -pad:, :], w // 2, -1), dims=[-2])
    x = torch.cat([top, x, bottom], dim=-2)
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


class ERPCircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(erp_pad(x, self.padding))


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = ERPCircularConv2d(features, features, 3)
        self.conv2 = ERPCircularConv2d(features, features, 3)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, x: Tensor, skip: Tensor = None, size=None) -> Tensor:
        if skip is not None:
            x = x + self.resConfUnit1(skip)
        x = self.resConfUnit2(x)
        kwargs = {"scale_factor": 2} if size is None else {"size": size}
        x = F.interpolate(x, mode="bilinear", align_corners=True, **kwargs)
        return self.out_conv(x)


class DPTHead(nn.Module):
    def __init__(self, in_channels: int, features: int, out_channels):
        super().__init__()
        oc = out_channels
        self.projects = nn.ModuleList(nn.Conv2d(in_channels, c, kernel_size=1) for c in oc)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(oc[0], oc[0], kernel_size=4, stride=4),
            nn.ConvTranspose2d(oc[1], oc[1], kernel_size=2, stride=2),
            nn.Identity(),
            ERPCircularConv2d(oc[3], oc[3], 3, stride=2),
        ])

        s = self.scratch = nn.Module()
        s.layer1_rn = ERPCircularConv2d(oc[0], features, 3, bias=False)
        s.layer2_rn = ERPCircularConv2d(oc[1], features, 3, bias=False)
        s.layer3_rn = ERPCircularConv2d(oc[2], features, 3, bias=False)
        s.layer4_rn = ERPCircularConv2d(oc[3], features, 3, bias=False)
        s.refinenet1 = FeatureFusionBlock(features)
        s.refinenet2 = FeatureFusionBlock(features)
        s.refinenet3 = FeatureFusionBlock(features)
        s.refinenet4 = FeatureFusionBlock(features)
        s.output_conv1 = ERPCircularConv2d(features, features // 2, 3)
        s.output_conv2 = nn.Sequential(
            ERPCircularConv2d(features // 2, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, features, patch_h: int, patch_w: int) -> Tensor:
        layers = []
        for i, (tokens, _cls_token) in enumerate(features):
            x = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
            layers.append(self.resize_layers[i](self.projects[i](x)))

        s = self.scratch
        l1 = s.layer1_rn(layers[0])
        l2 = s.layer2_rn(layers[1])
        l3 = s.layer3_rn(layers[2])
        l4 = s.layer4_rn(layers[3])

        p4 = s.refinenet4(l4, size=l3.shape[2:])
        p3 = s.refinenet3(p4, l3, size=l2.shape[2:])
        p2 = s.refinenet2(p3, l2, size=l1.shape[2:])
        p1 = s.refinenet1(p2, l1)

        out = s.output_conv1(p1)
        out = F.interpolate(out, (patch_h * PATCH_SIZE, patch_w * PATCH_SIZE), mode="bilinear", align_corners=True)
        return s.output_conv2(out)


# --------------------------------------------------------------------------- #
# Depth Anything V2 (backbone + DPT head)
# --------------------------------------------------------------------------- #
class DepthAnythingV2(nn.Module):
    def __init__(self, encoder: str):
        super().__init__()
        cfg = ENCODER_CONFIGS[encoder]
        self.pretrained = DinoVisionTransformer(cfg["embed_dim"], cfg["depth"], cfg["num_heads"], cfg["layer_ids"])
        self.depth_head = DPTHead(cfg["embed_dim"], cfg["features"], cfg["out_channels"])

    def forward(self, x: Tensor):
        patch_h, patch_w = x.shape[-2] // PATCH_SIZE, x.shape[-1] // PATCH_SIZE
        features = self.pretrained.get_intermediate_layers(x)
        depth = self.depth_head(features, patch_h, patch_w)
        cls_token = features[-1][1]
        return depth, cls_token


# --------------------------------------------------------------------------- #
# DA360
# --------------------------------------------------------------------------- #
class ShiftMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        hidden = [in_dim // 2, in_dim // 4]
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden[0]), nn.Linear(hidden[0], hidden[1])])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden[0]), nn.BatchNorm1d(hidden[1])])
        self.output_layer = nn.Linear(hidden[1], 1)

    def forward(self, x: Tensor) -> Tensor:
        for linear, norm in zip(self.layers, self.norms):
            x = F.relu(norm(linear(x)))
        return F.softplus(self.output_layer(x))


class DA360(nn.Module):
    EPS = 1e-4

    def __init__(self, equi_h=518, equi_w=1036, dinov2_encoder="vits"):
        super().__init__()
        if dinov2_encoder not in ENCODER_CONFIGS:
            raise ValueError(f"Unsupported DINOv2 encoder: {dinov2_encoder}")

        self.depth_anything = DepthAnythingV2(dinov2_encoder)
        self.shift_mlp = ShiftMLP(ENCODER_CONFIGS[dinov2_encoder]["out_channels"][-1])

    def forward(self, input_equi_image: Tensor) -> dict:
        ssidisp, cls_token = self.depth_anything(input_equi_image)
        shift = self.shift_mlp(cls_token).unsqueeze(-1).unsqueeze(-1) + self.EPS
        return {"pred_disp": ssidisp + shift}
