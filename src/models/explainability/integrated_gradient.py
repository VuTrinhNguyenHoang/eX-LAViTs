import torch
import torch.nn as nn

class IG(nn.Module):
    def __init__(self, model, steps=30, has_cls=True):
        super().__init__()
        self.model = model
        self.steps = steps
        self.has_cls = has_cls

    def attribute(self, x, y_true):
        self.model.eval()
        B = x.size(0)

        # baseline = zero image
        x0 = torch.zeros_like(x)

        grads = 0
        for i in range(1, self.steps+1):
            alpha = i/self.steps
            x_step = x0 + alpha*(x-x0)
            x_step.requires_grad_(True)

            logit = self.model(x_step).gather(1, y_true[:,None]).sum()
            self.model.zero_grad()
            logit.backward(retain_graph=True)

            # gradient wrt patch embeds
            with torch.no_grad():
                feats = self.model.patch_embed(x_step)  # [B,N,C]
                grad = x_step.grad

            grads += grad

        IG = (x - x0) * (grads / self.steps)

        # convert to token-level by summing channels
        rtokens = IG.sum(dim=1)   # [B,N]

        return {"rtokens": rtokens}