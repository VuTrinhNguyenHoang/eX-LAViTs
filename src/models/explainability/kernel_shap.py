import torch
import torch.nn as nn

class SHAP(nn.Module):
    def __init__(self, model, samples=50, has_cls=True):
        super().__init__()
        self.model = model
        self.samples = samples
        self.has_cls = has_cls

    @torch.no_grad()
    def attribute(self, x, y_true):
        self.model.eval()

        tokens = self.model.patch_embed(x)  # [B,N,C]
        B,N,C = tokens.shape
        base = self.model(x).gather(1, y_true[:,None])[:,0]  # [B]

        Ms = []
        Ys = []

        for _ in range(self.samples):
            m = torch.randint(0,2,(B,N), device=x.device)
            t = tokens * m[:,:,None]

            out = self.model.forward_features_from_tokens(t)
            score = out.gather(1, y_true[:,None])[:,0]

            Ms.append(m)
            Ys.append(score)

        M = torch.stack(Ms, dim=2).float()  # [B,N,S]
        Y = torch.stack(Ys, dim=1)          # [B,S]

        # simple least-squares shap
        MT = M.transpose(1,2)  # [B,S,N]
        A = MT.transpose(1,2) @ MT + 1e-6*torch.eye(N).to(x.device)
        b = MT.transpose(1,2) @ Y.unsqueeze(-1)
        phi = torch.linalg.solve(A, b).squeeze(-1)   # [B,N]

        return {"rtokens": phi.clamp(min=0)}