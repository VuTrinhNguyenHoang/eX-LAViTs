from ..linear_vit import LinearMultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple

def set_linear_attn_record(model: nn.Module, record: bool = True):
    for m in model.modules():
        if isinstance(m, LinearMultiheadAttention):
            m.record_attn = record
            m.attn_map = None

def tokens_to_heatmap(
    patch_rel: torch.Tensor,  # [B, N_patches]
    model: nn.Module,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Chuyển relevance patch-level → heatmap [B, 1, H, W] (align với ảnh input).
    """
    pe = getattr(model, "patch_embed", None)
    if pe is None or getattr(pe, "grid_size", None) is None:
        raise RuntimeError("model.patch_embed.grid_size không tồn tại.")

    Hp, Wp = pe.grid_size  # số patch theo chiều H, W
    B, Np = patch_rel.shape
    assert Np == Hp * Wp, f"Mismatch N_patches={Np}, Hp*Wp={Hp*Wp}"

    # [B, 1, Hp, Wp]
    attn_map = patch_rel.view(B, 1, Hp, Wp)

    # Lấy kích thước ảnh
    img_size = getattr(model, "img_size", None)
    if img_size is None:
        ps = getattr(pe, "patch_size", None)
        if isinstance(ps, (tuple, list)):
            ps_h, ps_w = ps
        else:
            ps_h = ps_w = int(ps)
        H = Hp * ps_h
        W = Wp * ps_w
    else:
        if isinstance(img_size, (tuple, list)):
            H, W = img_size
        else:
            H = W = int(img_size)

    heatmap = F.interpolate(
        attn_map,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # [B,1,H,W]

    if normalize:
        heatmap = heatmap - heatmap.amin(dim=(1, 2, 3), keepdim=True)
        heatmap = heatmap / (heatmap.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)

    return heatmap  # [B,1,H,W]

class LAGRA:
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        blocks_attr: str = "blocks",
        eps: float = 1e-6
    ):
        self.model = model
        self.model.eval()
        self.blocks: List[nn.Module] = list(getattr(model, blocks_attr))
        self.has_cls = has_cls
        self.eps = eps

    def _enable_record_attn(self, flag: bool):
        set_linear_attn_record(self.model, record=flag)

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        self.model.zero_grad(set_to_none=True)
        self._enable_record_attn(True)

        logits = self.model(x)  # [B,C]
        if target is None:
            target = logits.argmax(dim=-1)  # [B]

        one_hot = torch.zeros_like(logits).to(device)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        
        if use_logits:
            out = (logits * one_hot).sum()
        else:
            probs = logits.softmax(dim=-1)
            out = (probs * one_hot).sum()

        self.model.zero_grad(set_to_none=True)
        out.backward(retain_graph=False)

        R_all = None
        for blk in self.blocks:
            attn_layer = getattr(blk, "attn", None)
            if not isinstance(attn_layer, LinearMultiheadAttention):
                continue
            attn = attn_layer.attn_map
            if attn is None:
                continue
            grad = attn_layer.attn_map.grad
            if grad is None:
                continue

            joint = (attn * grad).clamp_min(0.0)  # [B,H,N,N]
            joint = joint.mean(dim=1)             # [B,N,N]

            joint_sum = joint.sum(dim=-1, keepdim=True) + self.eps
            joint = joint / joint_sum

            B_, N, _ = joint.shape
            eye = torch.eye(N, device=device).unsqueeze(0).expand(B_, -1, -1)
            joint = joint + eye
            joint = joint / (joint.sum(dim=-1, keepdim=True) + self.eps)

            if R_all is None:
                R_all = joint
            else:
                R_all = torch.bmm(R_all, joint)

        self._enable_record_attn(False)
        self.model.zero_grad(set_to_none=True)

        if R_all is None:
            raise RuntimeError("Không thu được attn_map nào trong LAAttributor.")

        if self.has_cls:
            cls_idx = 0
            token_rel = R_all[:, cls_idx, :]  # [B,N]
            patch_rel = token_rel[:, 1:]      # [B,N_p]
        else:
            patch_rel = R_all.mean(dim=1)

        patch_rel = patch_rel.clamp_min(0.0)
        patch_rel = patch_rel / (patch_rel.amax(dim=-1, keepdim=True) + self.eps)

        heatmap = tokens_to_heatmap(patch_rel, self.model, normalize=True)
        return patch_rel, heatmap
    
