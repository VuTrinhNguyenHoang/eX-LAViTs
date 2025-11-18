import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class IntegratedGradient(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        steps: int = 32,
        has_cls: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.steps = steps
        self.has_cls = has_cls
        self.eps = eps

        self.grid_hw: Optional[Tuple[int, int]] = getattr(
            getattr(model, "patch_embed", None), "grid_size", None
        )

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        self.model.eval()
        device = next(self.model.parameters()).device

        x = x.to(device)
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(device)

        # xác định target từ x thật, giữ cố định cho mọi bước
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        B, C = logits.shape

        if target is None:
            target = logits.argmax(dim=-1)
        else:
            target = target.to(logits.device)

        idx = torch.arange(B, device=logits.device)

        # tích lũy gradient
        total_grad = torch.zeros_like(x)

        for t in range(1, self.steps + 1):
            alpha = float(t) / self.steps
            x_t = baseline + alpha * (x - baseline)
            x_t = x_t.detach().requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            logits_t = self.model(x_t)

            logit_target_t = logits_t[idx, target]
            logit_target_t.sum().backward()

            total_grad += x_t.grad.detach()

        avg_grad = total_grad / float(self.steps)

        # IG attribution
        attributions = (x - baseline) * avg_grad  # [B, 3, H, W]

        # gộp kênh -> [B,1,H,W]
        attr_map = attributions.sum(dim=1, keepdim=True)

        # chuẩn hoá 0–1
        attr_map = attr_map - attr_map.amin(dim=(2, 3), keepdim=True)
        attr_map = attr_map / (
            attr_map.amax(dim=(2, 3), keepdim=True) + self.eps
        )

        # mapping về token_scores bằng pooling theo grid
        if self.grid_hw is not None:
            H_p, W_p = self.grid_hw
        else:
            # suy ra từ kích thước ảnh nếu chưa có
            _, _, H_img, W_img = attr_map.shape
            H_p, W_p = H_img // 16, W_img // 16

        # [B,1,H_p,W_p]
        patch_attr = F.adaptive_avg_pool2d(attr_map, (H_p, W_p))
        token_scores = patch_attr.reshape(x.shape[0], -1)  # [B, N_patches]

        # chuẩn hoá lại tokens
        token_scores = token_scores - token_scores.amin(dim=1, keepdim=True)
        token_scores = token_scores / (
            token_scores.amax(dim=1, keepdim=True) + self.eps
        )

        # heatmap ở resolution ảnh
        if img_size is not None:
            heatmap = F.interpolate(
                attr_map,
                size=img_size,
                mode="bilinear",
                align_corners=False,
            )
        else:
            heatmap = attr_map

        return token_scores, heatmap
