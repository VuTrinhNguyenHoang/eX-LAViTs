import torch
import torch.nn as nn

class LinearRollout(nn.Module):
    def __init__(self, model, alpha: float = 0.95, has_cls: bool = True, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.alpha = alpha
        self.blocks = model.blocks
        self.has_cls = has_cls
        self.eps = eps

    @torch.no_grad()
    def _attn_weights(self, blk, attn_in):
        attn = blk.attn
        B,N,C = attn_in.shape
        H = attn.h
        D = attn.d
        
        qkv = attn.qkv(attn_in).view(B,N,3,H,D).permute(2,0,3,1,4)
        qf = attn._phi(qkv[0])
        kf = attn._phi(qkv[1])
        
        s = torch.einsum("bhnd,bhmd->bhnm", qf, kf) + self.eps
        A = s / s.sum(dim=-1, keepdim=True)
        return A.mean(dim=1)   # average heads: [B,N,N]
    
    @torch.no_grad()
    def attribute(self, x, y_true=None):
        self.model.eval()
        B = x.size(0)

        # get token embeddings through each block
        attn_ins = []
        out = x
        for blk in self.blocks:
            x_in = blk.norm1(out)
            attn_ins.append(x_in)
            out = blk(out)

        # compute rollout
        A_roll = None
        for li, blk in enumerate(self.blocks):
            A = self._attn_weights(blk, attn_ins[li])   # [B,N,N]
            A = self.alpha*A + (1-self.alpha)*torch.eye(A.size(-1), device=x.device)

            if A_roll is None:
                A_roll = A
            else:
                A_roll = A @ A_roll

        # lấy token-level relevance: dòng CLS
        if self.has_cls:
            rtokens = A_roll[:,0]      # [B,N]
        else:
            rtokens = A_roll.mean(dim=1)

        return {"rtokens": rtokens}