import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

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

class Rollout(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.model = model
        self.blocks: List[nn.Module] = list(getattr(model, "blocks"))
        self.has_cls = has_cls
        self.eps = eps

        self.grid_hw: Optional[Tuple[int, int]] = getattr(
            getattr(model, "patch_embed", None), "grid_size", None
        )

    def _collect_attn(self) -> List[torch.Tensor]:
        attn_maps = []
        for blk in self.blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is None:
                raise RuntimeError("Block không có 'attn'.")
            attn = getattr(attn_mod, "attn_map", None)
            if attn is None:
                raise RuntimeError(
                    "block.attn.attn_map is None. "
                    "Hãy bật attn_mod.record_attn=True trước forward."
                )
            attn_maps.append(attn.detach())  # [B, H, N, N]
        return attn_maps

    def _build_rollout_matrix(self, attn_maps: List[torch.Tensor]) -> torch.Tensor:
        B, H, N, _ = attn_maps[0].shape
        device = attn_maps[0].device

        A_rollout = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)

        for A_l in attn_maps:
            # average over heads
            A_mean = A_l.mean(dim=1)  # [B, N, N]
            I = torch.eye(N, device=device).unsqueeze(0)  # [1, N, N]
            A_tilde = A_mean + I
            A_rollout = torch.bmm(A_rollout, A_tilde)

        return A_rollout  # [B, N, N]

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,  # không dùng, chỉ để API giống
        img_size: Optional[Tuple[int, int]] = None,
    ):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        # bật record_attn
        for blk in self.blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "record_attn"):
                attn_mod.record_attn = True

        # 1 forward là đủ
        with torch.no_grad():
            _ = self.model(x)

        attn_maps = self._collect_attn()
        A_rollout = self._build_rollout_matrix(attn_maps)

        cls_row = A_rollout[:, 0, :]  # [B, N]

        if self.has_cls:
            token_scores = cls_row[:, 1:]
        else:
            token_scores = cls_row

        # chuẩn hoá
        token_scores = token_scores - token_scores.amin(dim=1, keepdim=True)
        token_scores = token_scores / (
            token_scores.amax(dim=1, keepdim=True) + self.eps
        )

        heatmap = tokens_to_heatmap(token_scores, self.grid_hw, img_size, self.eps)
        return token_scores, heatmap