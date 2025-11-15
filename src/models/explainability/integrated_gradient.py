import torch
import torch.nn as nn

class IG(nn.Module):
    def __init__(self, model, steps=30, has_cls=True, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.steps = steps
        self.has_cls = has_cls and hasattr(model, "cls_token")
        self.eps = eps

    def _forward_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = tokens
        B, N, C = x.shape

        # 1) thêm CLS
        if self.has_cls:
            cls_tok = m.cls_token.expand(B, -1, -1)        # [B,1,C]
            x = torch.cat([cls_tok, x], dim=1)             # [B,1+N,C]

        # 2) pos_embed
        if hasattr(m, "pos_embed"):
            pe = m.pos_embed                               # [1,1+N,C] thường là vậy
            if pe.size(1) != x.size(1):
                pe = pe[:, :x.size(1), :]
            x = x + pe

        # 3) pos_drop
        if hasattr(m, "pos_drop"):
            x = m.pos_drop(x)

        # 4) blocks
        for blk in m.blocks:
            x = blk(x)

        # 5) norm + lấy cls / pooled
        if hasattr(m, "norm"):
            x = m.norm(x)

        if self.has_cls:
            feat = x[:, 0]                                 # [B,C]
        else:
            feat = x.mean(dim=1)

        # 6) head
        if hasattr(m, "head") and isinstance(m.head, nn.Linear):
            logits = m.head(feat)                          # [B,K]
        else:
            logits = feat
        return logits

    def attribute(self, x, y_true):
        device = x.device
        self.model.eval()

        x = x.to(device)
        y_true = y_true.view(-1).to(device)
        B = x.size(0)

        with torch.no_grad():
            tokens = self.model.patch_embed(x)             # [B,N,C]
        B, N, C = tokens.shape

        tokens0 = torch.zeros_like(tokens)
        grads = torch.zeros_like(tokens)

        for i in range(1, self.steps + 1):
            alpha = float(i) / self.steps
            t_step = tokens0 + alpha * (tokens - tokens0)  # [B,N,C]
            t_step.requires_grad_(True)

            logits = self._forward_from_tokens(t_step)     # [B,K]
            score = logits.gather(1, y_true[:, None]).sum()

            self.model.zero_grad(set_to_none=True)
            if t_step.grad is not None:
                t_step.grad.zero_()
            score.backward()

            grads += t_step.grad.detach()

        # IG trong không gian token
        ig_tokens = (tokens - tokens0) * (grads / self.steps)    # [B,N,C]

        # chuyển thành scalar relevance per token (sum over channels)
        rtokens = ig_tokens.sum(dim=-1)                          # [B,N]
        rtokens = rtokens.clamp_min(0.)

        return {"rtokens": rtokens}