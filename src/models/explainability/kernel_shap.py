import torch
import torch.nn as nn

class SHAP(nn.Module):
    def __init__(self, model, samples=50, has_cls=True, eps=1e-6):
        super().__init__()
        self.model = model
        self.samples = samples
        self.has_cls = has_cls and hasattr(model, "cls_token")
        self.eps = eps
    
    @torch.no_grad()
    def _forward_from_tokens(self, tokens: torch.Tensor):
        m = self.model
        x = tokens
        B, N, C = x.shape

        if self.has_cls:
            cls_token = m.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
        
        if hasattr(m, "pos_embed"):
            pe = m.pos_embed
            if pe.size(1) != x.size(1):
                pe = pe[:, : x.size(1), :]
            x = x + pe
        
        if hasattr(m, "pos_drop"):
            x = m.pos_drop(x)

        for blk in m.blocks:
            x = blk(x) 

        if hasattr(m, "norm"):
            x = m.norm(x)

        if self.has_cls:
            feat = x[:, 0] 
        else:
            feat = x.mean(dim=1)

        if hasattr(m, "head") and isinstance(m.head, nn.Linear):
            logits = m.head(feat)                          # [B,K]
        else:
            logits = feat
        
        return logits

    @torch.no_grad()
    def attribute(self, x, y_true):
        self.model.eval()
        device = x.device

        tokens = self.model.patch_embed(x)  # [B,N,C]
        B,N,C = tokens.shape
        
        y_true = y_true.view(-1).to(device)
        assert y_true.size(0) == B

        logits_full = self._forward_from_tokens(tokens)    # [B,K]
        score_full = logits_full.gather(1, y_true[:, None])[:, 0]  # [B]

        Ms = []
        Ys = []

        L = self.num_samples
        for _ in range(L):
            m = torch.randint(0, 2, (B, N), device=device, dtype=torch.float32)
            s = m.sum(dim=1, keepdim=True)

            all_zero = (s == 0)
            all_one  = (s == N)
            if all_zero.any():
                m[all_zero.squeeze(1)] = 1.0
            if all_one.any():
                m[all_one.squeeze(1)] = 0.0

            t = tokens * m.unsqueeze(-1)
            logits = self._forward_from_tokens(t)
            score = logits.gather(1, y_true[:, None])[:, 0]

            Ms.append(m)
            Ys.append(score)

        M = torch.stack(Ms, dim=1)  # [B,N,S]
        Y = torch.stack(Ys, dim=1)          # [B,S]

        S_card = M.sum(dim=-1)                             # [B,L]
        S_card = S_card.clamp_(1, N-1)                     # tránh 0,N
        w = (N - 1) / (S_card * (N - S_card))              # [B,L]

        phi_list = []
        eyeN = torch.eye(N, device=device)

        for b in range(B):
            Mb = M[b]                                      # [L,N]
            Yb = Y[b]                                      # [L]
            wb = w[b]                                      # [L]

            Wb = torch.diag(wb)                            # [L,L]
            A = Mb.t() @ Wb @ Mb + 1e-6 * eyeN             # [N,N]
            b_vec = Mb.t() @ (Wb @ Yb)                     # [N]

            phib = torch.linalg.solve(A, b_vec)            # [N]
            # clamp dương
            phib = phib.clamp_min(0.)
            phi_list.append(phib)

        phi = torch.stack(phi_list, dim=0)                 # [B,N]

        return {"rtokens": phi}