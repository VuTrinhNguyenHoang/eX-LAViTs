import torch
import torch.nn as nn

from typing import List

class LinearRollout(nn.Module):
    def __init__(self, model, alpha: float = 0.95, has_cls: bool = True, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.blocks: List[nn.Module] = list(model.blocks)
        self.has_cls = has_cls and hasattr(model, "cls_token")
        self.eps = eps

    @torch.no_grad()
    def _attn_weights_linear(self, blk, attn_in):
        attn = blk.attn
        B, N, C = attn_in.shape
        H = attn.h
        D = attn.d
        
        qkv = attn.qkv(attn_in).view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        qf = attn._phi(q)
        kf = attn._phi(k)
        
        s = torch.einsum("bhnd,bhmd->bhnm", qf, kf)   # [B,H,N,N]
        s = s.clamp_min(0.) + self.eps

        A = s / (s.sum(dim=-1, keepdim=True) + self.eps)
        return A.mean(dim=1)   # average heads: [B,N,N]
    
    @torch.no_grad()
    def _forward_and_collect_attn_in(self, x: torch.Tensor):
        m = self.model
        B = x.size(0)

        x_tokens = m.patch_embed(x)

        if self.has_cls:
            cls_tok = m.cls_token.expand(B, -1, -1)          # [B,1,C]
            x_tokens = torch.cat([cls_tok, x_tokens], dim=1) # [B,1+Np,C]

        if hasattr(m, "pos_embed"):
            pe = m.pos_embed
            if pe.size(1) != x_tokens.size(1):
                pe = pe[:, : x_tokens.size(1), :]
            x_tokens = x_tokens + pe

        if hasattr(m, "pos_drop"):
            x_tokens = m.pos_drop(x_tokens)

        attn_ins: List[torch.Tensor] = []
        h = x_tokens
        for blk in self.blocks:
            # pre-LN ViT: attn input = norm1(h)
            h_norm = blk.norm1(h)
            attn_ins.append(h_norm.detach())
            h = blk(h)

        return h, attn_ins

    @torch.no_grad()
    def attribute(self, x, y_true=None):
        self.model.eval()
        x = x.to(next(self.model.parameters()).device)

        x_out, attn_ins = self._forward_and_collect_attn_in(x)

        B, N_tokens, _ = x_out.shape
        L = len(self.blocks)
        assert len(attn_ins) == L

        A_roll = None  # [B,N,N]
        for li, blk in enumerate(self.blocks):
            A = self._attn_weights_linear(blk, attn_ins[li])   # [B,N,N]
            # A' = alpha*A + (1-alpha)*I
            I = torch.eye(N_tokens, device=A.device, dtype=A.dtype).unsqueeze(0)  # [1,N,N]
            A = self.alpha * A + (1. - self.alpha) * I
            if A_roll is None:
                A_roll = A
            else:
                # rollout: A_l * A_{l-1} * ... * A_1
                A_roll = A @ A_roll

        # lấy token-level relevance: dòng CLS
        if self.has_cls:
            rtokens = A_roll[:, 0, :]      # [B,N]
        else:
            rtokens = A_roll.mean(dim=1)

        rtokens = rtokens.clamp_min(0.)
        return {"rtokens": rtokens}