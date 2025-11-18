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

class KernelSHAP(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 256,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.eps = eps

        self.grid_hw: Optional[Tuple[int, int]] = getattr(
            getattr(model, "patch_embed", None), "grid_size", None
        )

    def _build_masked_input(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x, baseline: [1, 3, H, W]
        mask: [N_patches]
        Trả về masked_x: [1, 3, H, W]
        """
        B, C, H, W = x.shape
        N_patches = mask.numel()

        if self.grid_hw is not None:
            H_p, W_p = self.grid_hw
        else:
            H_p = W_p = int(N_patches ** 0.5)

        patch_h = H // H_p
        patch_w = W // W_p

        mask_2d = mask.view(H_p, W_p)  # [H_p, W_p]

        masked_x = baseline.clone()

        # broadcast mask lên ảnh
        for i in range(H_p):
            for j in range(W_p):
                if mask_2d[i, j] == 1:
                    h0 = i * patch_h
                    h1 = (i + 1) * patch_h
                    w0 = j * patch_w
                    w1 = (j + 1) * patch_w
                    masked_x[:, :, h0:h1, w0:w1] = x[:, :, h0:h1, w0:w1]

        return masked_x

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
        assert (
            x.shape[0] == 1
        ), "KernelSHAPViT hiện chỉ hỗ trợ B=1 mỗi lần để đơn giản."

        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(device)

        # xác định target từ x thật
        with torch.no_grad():
            logits = self.model(x)  # [1, C]
        if target is None:
            target = logits.argmax(dim=-1)  # [1]
        else:
            target = target.to(logits.device)

        # số patch
        if self.grid_hw is not None:
            H_p, W_p = self.grid_hw
            N_p = H_p * W_p
        else:
            _, _, H, W = x.shape
            H_p = W_p = H // 16
            N_p = H_p * W_p

        # sinh mask
        M = self.n_samples
        Z = torch.zeros(M, N_p, device=device)
        y = torch.zeros(M, device=device)

        for m in range(M):
            # mask có ít nhất 1 patch on và không trivially all-ones
            while True:
                mask = torch.randint(0, 2, (N_p,), device=device)
                if mask.sum() > 0 and mask.sum() < N_p:
                    break

            masked_x = self._build_masked_input(x, baseline, mask).to(device)

            with torch.no_grad():
                logits_m = self.model(masked_x)  # [1, C]

            y[m] = logits_m[0, target.item()]
            Z[m] = mask

        # thêm cột intercept
        # design matrix: [M, N_p + 1]
        ones = torch.ones(M, 1, device=device)
        X_mat = torch.cat([ones, Z], dim=1)  # [M, N_p + 1]

        # giải least squares: (X^T X)^{-1} X^T y
        # beta: [N_p + 1]
        beta, *_ = torch.linalg.lstsq(X_mat, y.unsqueeze(-1))  # [N_p + 1, 1]
        beta = beta.squeeze(-1)

        # shap patch = beta[1:]
        shap_vals = beta[1:]  # [N_p]

        # chuẩn hoá 0–1
        shap_vals = shap_vals.unsqueeze(0)  # [1, N_p]
        shap_vals = shap_vals - shap_vals.amin(dim=1, keepdim=True)
        shap_vals = shap_vals / (shap_vals.amax(dim=1, keepdim=True) + self.eps)

        token_scores = shap_vals  # [1, N_p]
        heatmap = tokens_to_heatmap(token_scores, self.grid_hw, img_size, self.eps)

        return token_scores, heatmap
