import torch
import torch.nn as nn

class Occlusion(nn.Module):
    def __init__(self, model: nn.Module, has_cls: bool = True):
        super().__init__()
        self.model = model
        self.has_cls = has_cls

    @torch.no_grad()
    def attribute(self, x, y_true):
        self.model.eval()
        B = x.size(0)

        # baseline score
        base = self.model(x).gather(1, y_true[:,None])[:,0]  # [B]

        # get patch embeddings
        tokens = self.model.patch_embed(x)      # [B,N,C]
        B,N,C = tokens.shape

        rtokens = torch.zeros(B,N, device=x.device)

        for j in range(N):
            t_masked = tokens.clone()
            t_masked[:,j] = 0

            out = self.model.forward_features_from_tokens(t_masked)
            logit = out.gather(1, y_true[:,None])[:,0]

            rtokens[:,j] = (base - logit).clamp(min=0)

        return {"rtokens": rtokens}