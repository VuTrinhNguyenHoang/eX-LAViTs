from typing import Optional, Literal, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import build_transformer_encoder

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
        
    def forward(self, x):
        x = self.proj(x)                       # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)       # [B, N, D]
        return x
    
class _ViTCore(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        qkv_bias: bool = True,
        act: Literal["gelu", "relu"] = "gelu",
        attn_type: Literal["sdpa", "linear"] = "sdpa",
        kernel: Literal["elu", "relu"] = "elu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))

        self.encoder = build_transformer_encoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            proj_drop=proj_drop_rate,
            qkv_bias=qkv_bias,
            activation=act,
            attn_type=attn_type,
            kernel=kernel,
            layer_norm_eps=layer_norm_eps,
        )

        self.norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,C,H,W]
        B = x.size(0)
        x = self.patch_embed(x)                # [B,N,C]
        cls = self.cls_token.expand(B, -1, -1) # [B,1,C]
        x = torch.cat([cls, x], dim=1)         # [B,1+N,C]

        x = x + self.pos_embed

        x = self.encoder(x)                    # [B,1+N,C]
        x = self.norm(x)
        cls_out = x[:, 0]
        return cls_out, x[:, 1:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)

        return self.head(x[:, 0])
    
class StandardViT(_ViTCore):
    def __init__(self, **kwargs):
        super().__init__(attn_type="sdpa", **kwargs)

class LAViT(_ViTCore):
    def __init__(self, kernel: Literal["elu", "relu"] = "elu", **kwargs):
        super().__init__(attn_type="linear", kernel=kernel, **kwargs)
