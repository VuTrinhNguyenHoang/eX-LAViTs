import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .linear_rollout import tokens_to_heatmap, set_linear_attn_record
from ..linear_vit import LinearMultiheadAttention

class Rollout:
    """
    Attention Rollout cho ViT Linear Attention (không dùng gradient).
    """
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        blocks_attr: str = "blocks",
        eps: float = 1e-6,
    ):
        self.model = model
        self.model.eval()
        self.blocks: List[nn.Module] = list(getattr(model, blocks_attr))
        self.has_cls = has_cls
        self.eps = eps

    @torch.no_grad()
    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về:
            patch_rel: [B, N_patches]
            heatmap:  [B, 1, H, W]
        """
        device = x.device
        # bật record_attn để đi qua nhánh softmax-like
        set_linear_attn_record(self.model, True)

        logits = self.model(x)  # forward duy nhất

        R_all = None
        for blk in self.blocks:
            attn_layer = getattr(blk, "attn", None)
            if not isinstance(attn_layer, LinearMultiheadAttention):
                continue
            attn = attn_layer.attn_map  # [B,H,N,N]
            if attn is None:
                continue

            # trung bình head
            A = attn.mean(dim=1)  # [B,N,N]

            # normalize + thêm identity (skip)
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            B_, N, _ = A.shape
            eye = torch.eye(N, device=device).unsqueeze(0).expand(B_, -1, -1)
            A = A + eye
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)

            if R_all is None:
                R_all = A
            else:
                R_all = torch.bmm(R_all, A)

        # tắt record_attn
        set_linear_attn_record(self.model, False)

        if R_all is None:
            raise RuntimeError("Không thu được attn_map nào cho Rollout.")

        if self.has_cls:
            cls_idx = 0
            token_rel = R_all[:, cls_idx, :]     # [B,N]
            patch_rel = token_rel[:, 1:]         # [B,N_p]
        else:
            patch_rel = R_all.mean(dim=1)        # [B,N]

        # chuẩn hoá
        patch_rel = patch_rel / (patch_rel.amax(dim=-1, keepdim=True) + self.eps)

        heatmap = tokens_to_heatmap(patch_rel, self.model, normalize=True)
        return patch_rel, heatmap
    
class TextRollout:
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        blocks_attr: str = "blocks",
        eps: float = 1e-6,
    ):
        self.model = model
        self.model.eval()
        self.blocks: List[nn.Module] = list(getattr(model, blocks_attr))
        self.has_cls = has_cls
        self.eps = eps

    @torch.no_grad()
    def attribute(
        self,
        input_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> torch.Tensor:
        device = input_ids.device
        set_linear_attn_record(self.model, True)

        logits = self.model(input_ids)        # forward duy nhất

        R_all = None
        for blk in self.blocks:
            attn_layer = getattr(blk, "attn", None)
            if not isinstance(attn_layer, LinearMultiheadAttention):
                continue
            attn = attn_layer.attn_map        # [B,H,N,N]
            if attn is None:
                continue

            # trung bình head
            A = attn.mean(dim=1)              # [B,N,N]

            # normalize + thêm identity (skip)
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            B_, N, _ = A.shape
            eye = torch.eye(N, device=device).unsqueeze(0).expand(B_, -1, -1)
            A = A + eye
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)

            if R_all is None:
                R_all = A
            else:
                R_all = torch.bmm(R_all, A)    # [B,N,N]

        set_linear_attn_record(self.model, False)

        if R_all is None:
            raise RuntimeError("Không thu được attn_map nào cho TextRollout.")

        if self.has_cls:
            cls_idx = 0
            token_rel_full = R_all[:, cls_idx, :]  # [B,N]
            token_rel = token_rel_full[:, 1:]      # [B,L_tok]
        else:
            token_rel = R_all.mean(dim=1)          # [B,N]

        token_rel = token_rel / (token_rel.amax(dim=-1, keepdim=True) + self.eps)
        token_rel_map = token_rel.unsqueeze(1)     # [B,1,L_tok]

        return token_rel, token_rel_map