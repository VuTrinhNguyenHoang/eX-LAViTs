import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class IntegratedGradient:
    """
    Integrated Gradients trên ảnh input, sau đó average-pool về patch grid.
    """
    def __init__(
        self,
        model: nn.Module,
        m_steps: int = 32,
        baseline: Optional[torch.Tensor] = None,  # nếu None -> zeros
        eps: float = 1e-6,
    ):
        self.model = model
        self.model.eval()
        self.m_steps = int(m_steps)
        self.baseline = baseline
        self.eps = eps

        pe = getattr(self.model, "patch_embed", None)
        if pe is None or getattr(pe, "grid_size", None) is None:
            raise RuntimeError("Model không có patch_embed.grid_size cho IG patch-level.")
        self.Hp, self.Wp = pe.grid_size

    def _get_baseline(self, x: torch.Tensor) -> torch.Tensor:
        if self.baseline is not None:
            return self.baseline.to(x.device).expand_as(x)
        return torch.zeros_like(x)

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
            patch_rel: [B, N_patches]
            heatmap:  [B,1,H,W]
        """
        device = x.device
        B, C, H, W = x.shape
        baseline = self._get_baseline(x)

        self.model.zero_grad(set_to_none=True)

        # forward 1 lần để lấy target mặc định
        with torch.no_grad():
            logits = self.model(x)
        if target is None:
            target = logits.argmax(dim=-1)

        # get one_hot
        one_hot_template = torch.zeros_like(logits)

        # accumulate gradients
        total_grad = torch.zeros_like(x)

        for k in range(1, self.m_steps + 1):
            alpha = float(k) / self.m_steps
            x_k = baseline + alpha * (x - baseline)
            x_k.requires_grad_(True)

            logits_k = self.model(x_k)

            one_hot = one_hot_template.clone()
            one_hot.scatter_(1, target.view(-1, 1), 1.0)
            if use_logits:
                out_k = (logits_k * one_hot).sum()
            else:
                probs_k = logits_k.softmax(dim=-1)
                out_k = (probs_k * one_hot).sum()

            self.model.zero_grad(set_to_none=True)
            out_k.backward(retain_graph=False)

            grad_k = x_k.grad  # [B,3,H,W]
            total_grad += grad_k

        avg_grad = total_grad / self.m_steps
        ig = (x - baseline) * avg_grad  # [B,3,H,W]

        # scalar attribution per pixel: sum |IG| trên kênh
        heatmap = ig.abs().sum(dim=1, keepdim=True)  # [B,1,H,W]

        # chuẩn hoá
        heatmap = heatmap - heatmap.amin(dim=(1, 2, 3), keepdim=True)
        heatmap = heatmap / (heatmap.amax(dim=(1, 2, 3), keepdim=True) + self.eps)

        # average-pool về patch grid
        patch_map = F.adaptive_avg_pool2d(heatmap, (self.Hp, self.Wp))  # [B,1,Hp,Wp]
        patch_rel = patch_map.view(B, -1)  # [B,N_p]

        # chuẩn lại patch_rel
        patch_rel = patch_rel / (patch_rel.amax(dim=-1, keepdim=True) + self.eps)

        return patch_rel, heatmap
    
