import torch
import torch.nn as nn

class Occlusion(nn.Module):
    def __init__(self, model: nn.Module, has_cls: bool = True):
        super().__init__()
        self.model = model
        self.has_cls = has_cls and hasattr(model, "cls_token")

    @torch.no_grad()
    def _forward_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = tokens  # [B,N,C]
        B, N, C = x.shape

        # 1) thêm CLS nếu có
        if self.has_cls:
            cls_token = m.cls_token.expand(B, -1, -1)      # [B,1,C]
            x = torch.cat([cls_token, x], dim=1)           # [B,1+N,C]

        # 2) cộng pos_embed nếu có
        if hasattr(m, "pos_embed"):
            pe = m.pos_embed                               # [1, 1+N, C] (thường)
            if pe.size(1) != x.size(1):
                pe = pe[:, : x.size(1), :]
            x = x + pe

        # 3) dropout vị trí nếu có
        if hasattr(m, "pos_drop"):
            x = m.pos_drop(x)

        # 4) qua các blocks
        for blk in m.blocks:
            x = blk(x)                                     # [B,1+N,C]

        # 5) norm + lấy CLS (hoặc pooled)
        if hasattr(m, "norm"):
            x = m.norm(x)

        if self.has_cls:
            feat = x[:, 0]                                 # [B,C]
        else:
            feat = x.mean(dim=1)                           # [B,C]

        # 6) head → logits
        if hasattr(m, "head") and isinstance(m.head, nn.Linear):
            logits = m.head(feat)                          # [B,K]
        else:
            logits = feat

        return logits

    @torch.no_grad()
    def attribute(self, x, y_true):
        self.model.eval()
        device = x.device
        
        x = x.to(device)
        y_true = y_true.view(-1).to(device)
        B = x.size(0)

        # baseline score
        logits_full = self.model(x)                        # [B,K]
        base = logits_full.gather(1, y_true[:, None])[:, 0]  # [B]

        # get patch embeddings
        tokens = self.model.patch_embed(x)                 # [B,N,C]
        B, N, C = tokens.shape

        rtokens = torch.zeros(B, N, device=device)

        for j in range(N):
            t_masked = tokens.clone()
            t_masked[:, j] = 0

            logits_masked = self._forward_from_tokens(t_masked)  # [B,K]
            logit = logits_masked.gather(1, y_true[:, None])[:, 0]  # [B]

            rtokens[:, j] = (base - logit).clamp(min=0.)

        return {"rtokens": rtokens}