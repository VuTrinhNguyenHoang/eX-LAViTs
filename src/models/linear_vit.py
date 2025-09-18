import torch
import torch.nn as nn
import timm
from typing import Literal, Optional
from .transformer import LinearMultiheadAttention

class LinearAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        kernel: Literal["elu", "relu"] = "elu",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        # Thay standard attention bằng LinearMultiheadAttention
        self.attn = LinearMultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_bias=qkv_bias,
            kernel=kernel,
            eps=eps,
        )
        
        # Drop path for stochastic depth
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm like timm's ViT
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        bias: bool = True,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

def replace_attention_layers(
    model: nn.Module,
    kernel: Literal["elu", "relu"] = "elu",
    eps: float = 1e-6,
    verbose: bool = True
) -> nn.Module:
    """
    Thay thế tất cả attention layers trong pretrained ViT bằng LinearMultiheadAttention
    
    Args:
        model: Pretrained ViT model từ timm
        kernel: Kernel function cho linear attention ('elu' hoặc 'relu')
        eps: Epsilon value cho numerical stability
        verbose: In thông tin về quá trình thay thế
    
    Returns:
        Model với attention layers đã được thay thế
    """
    if verbose:
        print("Replacing attention layers with LinearMultiheadAttention...")
        print(f"    Kernel: {kernel}")
        print(f"    Epsilon: {eps}")
    
    replaced_count = 0
    
    # Tìm và thay thế blocks trong ViT
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn') and hasattr(block, 'norm1'):
                # Lấy thông tin từ block hiện tại
                dim = block.norm1.weight.shape[0]  # embed_dim
                num_heads = block.attn.num_heads
                qkv_bias = block.attn.qkv.bias is not None
                
                # Lấy dropout rates
                attn_drop = block.attn.attn_drop.p if hasattr(block.attn, 'attn_drop') else 0.0
                proj_drop = block.attn.proj_drop.p if hasattr(block.attn, 'proj_drop') else 0.0
                drop_path = block.drop_path.drop_prob if hasattr(block, 'drop_path') else 0.0
                mlp_drop = block.mlp.drop.p if hasattr(block.mlp, 'drop') else 0.0
                
                # Tính mlp_ratio từ hidden size
                mlp_ratio = block.mlp.fc1.out_features / dim if hasattr(block.mlp, 'fc1') else 4.0
                
                if verbose:
                    print(f"   Block {i}: dim={dim}, heads={num_heads}, mlp_ratio={mlp_ratio:.1f}")
                
                # Tạo LinearAttentionBlock mới
                new_block = LinearAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=mlp_drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path,
                    kernel=kernel,
                    eps=eps,
                )
                
                # Copy weights từ block cũ (trừ attention)
                # Copy norm layers
                new_block.norm1.load_state_dict(block.norm1.state_dict())
                new_block.norm2.load_state_dict(block.norm2.state_dict())
                
                # Copy MLP weights
                if hasattr(block.mlp, 'fc1') and hasattr(block.mlp, 'fc2'):
                    new_block.mlp.fc1.load_state_dict(block.mlp.fc1.state_dict())
                    new_block.mlp.fc2.load_state_dict(block.mlp.fc2.state_dict())
                
                # Copy drop_path if exists
                if hasattr(block, 'drop_path') and hasattr(new_block, 'drop_path1'):
                    new_block.drop_path1.drop_prob = block.drop_path.drop_prob
                    new_block.drop_path2.drop_prob = block.drop_path.drop_prob
                
                # Thay thế block
                model.blocks[i] = new_block
                replaced_count += 1
    
    if verbose:
        print(f"Successfully replaced {replaced_count} attention blocks")
        
        # Thống kê parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")
    
    return model

def create_linear_attention_vit(
    model_name: str = "vit_small_patch16_224",
    pretrained: bool = True,
    num_classes: int = 1000,
    img_size: int = 224,
    kernel: Literal["elu", "relu"] = "elu",
    eps: float = 1e-6,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    verbose: bool = True,
    **kwargs
) -> nn.Module:
    if verbose:
        print(f"Creating Linear Attention ViT: {model_name}")
        print(f"    Pretrained: {pretrained}")
        print(f"    Classes: {num_classes}")
        print(f"    Image size: {img_size}")
    
    # Tạo base model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    if verbose:
        original_params = sum(p.numel() for p in model.parameters())
        print(f"    Original parameters: {original_params:,}")
    
    # Thay thế attention layers
    model = replace_attention_layers(
        model=model,
        kernel=kernel,
        eps=eps,
        verbose=verbose
    )
    
    return model
