import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

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

class ViTGradCAM(nn.Module):
    def __init__(self, model: nn.Module, has_cls: bool = True, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.has_cls = has_cls
        self.eps = eps

        self.blocks: List[nn.Module] = list(getattr(model, "blocks"))
        self.grid_hw: Optional[Tuple[int, int]] = getattr(
            getattr(model, "patch_embed", None), "grid_size", None
        )

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        last_block = self.blocks[-1]

        def f_hook(module, inp, out):
            self.activations = out  # [B, N, C]

        def b_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]  # [B, N, C]

        last_block.register_forward_hook(f_hook)
        last_block.register_full_backward_hook(b_hook)

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        x = x.to(next(self.model.parameters()).device)
        logits = self.model(x)  # [B, C]
        B, C = logits.shape

        if target is None:
            target = logits.argmax(dim=-1)
        else:
            target = target.to(logits.device)

        idx = torch.arange(B, device=logits.device)
        logit_target = logits[idx, target]

        logit_target.sum().backward()

        A = self.activations  # [B, N, C]
        G = self.gradients    # [B, N, C]

        if A is None or G is None:
            raise RuntimeError("Hooks của GradCAM chưa được kích hoạt.")

        # trọng số kênh: mean gradient trên tokens
        # w: [B, C]
        w = G.mean(dim=1)

        # CAM per token: Σ_k w_k * A_{j,k}
        # [B, N]
        cam = (A * w.unsqueeze(1)).sum(dim=-1)
        cam = F.relu(cam)

        # bỏ cls token nếu có
        if self.has_cls:
            token_scores = cam[:, 1:]
        else:
            token_scores = cam

        # chuẩn hoá mỗi mẫu
        token_scores = token_scores - token_scores.amin(dim=1, keepdim=True)
        token_scores = token_scores / (
            token_scores.amax(dim=1, keepdim=True) + self.eps
        )

        heatmap = tokens_to_heatmap(token_scores, self.grid_hw, img_size, self.eps)
        return token_scores, heatmap
    
