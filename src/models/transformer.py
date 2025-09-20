from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        kernel: Literal["elu", "relu"] = "elu",
        eps: float = 1e-6,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.kernel = kernel
        self.eps = eps

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure non-negativity for kernel feature maps
        if self.kernel == "elu":
            return F.elu(x, alpha=1.0) + 1.0
        elif self.kernel == "relu":
            return F.relu(x)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.h, self.d)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,D]

        qf = self._phi(q)                  # [B,H,N,D]
        kf = self._phi(k)                  # [B,H,N,D]
        kf = self.attn_drop(kf)

        # kv term: [B,H,D,D]
        kv = torch.matmul(kf.float().transpose(-1, -2), v.float())
        kv = kv.to(v.dtype)

        # numerator: [B,H,N,D]
        y_num = torch.matmul(qf, kv)

        # denominator: [B,H,N,1]
        z = kf.float().sum(dim=2)                          # [B,H,D]
        y_den = torch.einsum("bhnd,bhd->bhn", qf, z.to(qf.dtype))  # [B,H,N]
        y_den = y_den.unsqueeze(-1).clamp_min(self.eps)    # [B,H,N,1]

        # output: [B,N,C]
        y = (y_num / y_den).transpose(1, 2).reshape(B, N, C)
        y = self.out_proj(y)
        y = self.proj_drop(y)

        return y
    
class LinearTransformerEncoderLayer(nn.Module):
    """A custom TransformerEncoderLayer that uses LinearMultiheadAttention"""
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        # Linear attention specific params
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
        kernel: Literal["elu", "relu"] = "elu",
    ):
        super().__init__()
        self.self_attn = LinearMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            kernel=kernel
        )
        
        # Feed forward layers
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
        self.norm_first = norm_first
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if self.norm_first:
            # Pre-norm
            attn_out = self.self_attn(self.norm1(src))
            src = src + self.dropout1(attn_out)
            
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
            src = src + self.dropout2(ff_out)
        else:
            # Post-norm
            attn_out = self.self_attn(src)
            src = self.norm1(src + self.dropout1(attn_out))
            
            ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = self.norm2(src + self.dropout2(ff_out))
            
        return src

def build_encoder_layer(
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    drop: float = 0.0,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    qkv_bias: bool = True,
    activation: Literal["gelu", "relu"] = "gelu",
    attn_type: Literal["sdpa", "linear"] = "sdpa",
    # tham số cho Linear Attention
    kernel: Literal["elu", "relu"] = "elu",
):
    if attn_type == "linear":
        layer = LinearTransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation=activation,
            norm_first=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            kernel=kernel
        )
    else:
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
    
    return layer

def build_transformer_encoder(
    depth: int,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    drop: float = 0.0,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    qkv_bias: bool = True,
    activation: Literal["gelu", "relu"] = "gelu",
    attn_type: Literal["sdpa", "linear"] = "sdpa",
    layer_norm_eps: float = 1e-5,
    # tham số cho Linear Attention
    kernel: Literal["elu", "relu"] = "elu",
):
    if attn_type == "linear":
        # Create custom encoder with LinearTransformerEncoderLayer
        layers = [
            LinearTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                norm_first=True,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                qkv_bias=qkv_bias,
                kernel=kernel
            ) for _ in range(depth)
        ]
        
        class _Encoder(nn.Module):
            def __init__(self, layers):
                super().__init__()
                self.layers = nn.ModuleList(layers)
            
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x)
                return x
            
        return _Encoder(layers)
    else:
        # Use standard PyTorch TransformerEncoder
        layer = build_encoder_layer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            qkv_bias=qkv_bias,
            activation=activation,
            attn_type=attn_type,
            kernel=kernel,
        )

        encoder = nn.TransformerEncoder(layer, num_layers=depth)
        for m in encoder.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = layer_norm_eps
        
        return encoder