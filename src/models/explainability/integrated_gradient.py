import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

class IntegratedGradient(nn.Module):
    def __init__(self, model: nn.Module, steps: int = 32):
        super().__init__()
        self.model = model
        self.steps = int(steps)

    def attribute(self,
                  x: torch.Tensor,
                  y_true: Optional[torch.Tensor] = None,
                  baseline: Optional[torch.Tensor] = None,
                  steps: Optional[int] = None):
        assert x.dim() == 4 and x.size(1) == 3
        B, _, H, W = x.shape
        steps = int(steps or self.steps)

        self.model.eval()
        torch.set_grad_enabled(True)
        for p in self.model.parameters():
            p.requires_grad_(True)

        if baseline is None:
            baseline = torch.zeros_like(x)

        if y_true is None:
            with torch.no_grad():
                logits = self.model(x)
                y_true = logits.argmax(dim=1)

        delta = x - baseline
        total_grad = torch.zeros_like(x)

        for i in range(1, steps + 1):
            alpha = float(i) / steps
            x_i = baseline + alpha * delta
            x_i.requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(x_i)
            score = logits.gather(1, y_true[:, None]).sum()
            score.backward()

            total_grad += x_i.grad.detach()

        avg_grad = total_grad / steps
        ig = delta * avg_grad         # [B,3,H,W]
        heat = ig.sum(dim=1)          # [B,H,W]
        return {"rpix": ig.detach(), "rtokens_up": heat.detach()}