from .linear_vit import LinearAttentionBlock
import torch
import torch.nn as nn
from typing import Literal

class LinearTextTransformer(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, max_seq_len: int = 512, embed_dim: int = 768,
                 depth: int = 6, num_heads: int = 12, mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.1, attn_drop_rate: float = 0.1, drop_path_rate: float = 0.1,
                 mlp_drop_rate: float = 0.1, kernel: Literal["elu", "relu"] = "elu", eps: float = 1e-6,
                 norm_layer: nn.Module = nn.LayerNorm, act_layer: nn.Module = nn.GELU):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            LinearAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                mlp_drop=mlp_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                kernel=kernel,
                eps=eps,
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)
    
    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)

    def forward_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        assert L <= self.max_seq_len, f"Input sequence length {L} exceeds maximum {self.max_seq_len}"

        x = self.token_embed(input_ids)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, L+1, embed_dim)
        
        x = x + self.pos_embed[:, :L + 1, :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        cls_out = x[:, 0]  # (B, embed_dim)
        return cls_out
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cls_out = self.forward_features(input_ids)
        logits = self.head(cls_out)
        return logits
    
