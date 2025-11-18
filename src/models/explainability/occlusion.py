import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

def tokens_to_heatmap(
    token_scores: torch.Tensor,
    grid_hw: Optional[Tuple[int, int]],
    img_size: Optional[Tuple[int, int]] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    token_scores: [B, N_patches] (không gồm cls)
    grid_hw: (H_p, W_p) hoặc None
    return: [B, 1, H_img, W_img]
    """
    B, num_patches = token_scores.shape

    if grid_hw is not None:
        H_p, W_p = grid_hw
    else:
        side = int(num_patches ** 0.5)
        H_p, W_p = side, side

    assert H_p * W_p == num_patches, "Số patch không khớp grid_size."

    maps = token_scores.reshape(B, 1, H_p, W_p)

    # chuẩn hoá 0–1
    maps = maps - maps.amin(dim=(2, 3), keepdim=True)
    maps = maps / (maps.amax(dim=(2, 3), keepdim=True) + eps)

    if img_size is not None:
        H_img, W_img = img_size
    else:
        # mặc định: mỗi patch 16×16 (vit_small_patch16_224)
        H_img, W_img = H_p * 16, W_p * 16

    maps_up = F.interpolate(
        maps, size=(H_img, W_img), mode="bilinear", align_corners=False
    )
    return maps_up


class Occlusion(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.eps = eps

        self.grid_hw: Optional[Tuple[int, int]] = getattr(
            getattr(model, "patch_embed", None), "grid_size", None
        )

    def _occlude_patch(
        self,
        x: torch.Tensor,
        baseline_val: float,
        patch_idx: int,
    ) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        patch_idx: index patch (0..N_p-1)
        """
        B, C, H, W = x.shape

        if self.grid_hw is not None:
            H_p, W_p = self.grid_hw
        else:
            H_p = W_p = int((H * W) ** 0.5)  # fallback (không khuyến nghị)

        patch_h = H // H_p
        patch_w = W // W_p

        i = patch_idx // W_p
        j = patch_idx % W_p

        h0, h1 = i * patch_h, (i + 1) * patch_h
        w0, w1 = j * patch_w, (j + 1) * patch_w

        x_masked = x.clone()
        x_masked[:, :, h0:h1, w0:w1] = baseline_val

        return x_masked

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        img_size: Optional[Tuple[int, int]] = None,
        baseline_val: float = 0.0,
    ):
        self.model.eval()
        device = next(self.model.parameters()).device

        x = x.to(device)
        B, C, H, W = x.shape

        with torch.no_grad():
            logits = self.model(x)  # [B, C]

        if target is None:
            target = logits.argmax(dim=-1)
        else:
            target = target.to(logits.device)

        idx = torch.arange(B, device=logits.device)
        logit_orig = logits[idx, target]  # [B]

        if self.grid_hw is not None:
            H_p, W_p = self.grid_hw
        else:
            H_p, W_p = H // 16, W // 16

        N_p = H_p * W_p
        scores = torch.zeros(B, N_p, device=device)

        # lặp patch
        for j in range(N_p):
            x_masked = self._occlude_patch(x, baseline_val, j)
            with torch.no_grad():
                logits_masked = self.model(x_masked)
            logit_m = logits_masked[idx, target]
            # importance = drop logit
            scores[:, j] = (logit_orig - logit_m)

        # chuẩn hoá mỗi mẫu
        scores = scores - scores.amin(dim=1, keepdim=True)
        scores = scores / (scores.amax(dim=1, keepdim=True) + self.eps)

        token_scores = scores
        heatmap = tokens_to_heatmap(token_scores, self.grid_hw, img_size, self.eps)
        return token_scores, heatmap
    
