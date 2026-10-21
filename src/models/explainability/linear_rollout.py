from ..linear_vit import LinearMultiheadAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Optional, Tuple
from dataclasses import dataclass

def set_linear_attn_record(model: nn.Module, record: bool = True):
    for m in model.modules():
        if isinstance(m, LinearMultiheadAttention):
            m.record_attn = record
            m.attn_map = None

def tokens_to_heatmap(
    token_rel: torch.Tensor,  # [B, N_tokens]
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
    B, Np = token_rel.shape
    assert Np == Hp * Wp, f"Mismatch N_patches={Np}, Hp*Wp={Hp*Wp}"

    # [B, 1, Hp, Wp]
    attn_map = token_rel.view(B, 1, Hp, Wp)

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

@dataclass
class LAGRAConfig:
    use_grad: bool = True
    use_relu: bool = True
    use_residual: bool = True
    normalize: str = "double"
    rollout: bool = True


class LAGRA:
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        blocks_attr: str = "blocks",
        eps: float = 1e-6,
        config: Optional[LAGRAConfig] = None,
    ):
        self.model = model.eval()
        self.blocks: List[nn.Module] = list(getattr(model, blocks_attr))
        self.has_cls = has_cls
        self.eps = eps
        self.cfg = config if config is not None else LAGRAConfig()

    def _enable_record_attn(self, flag: bool):
        set_linear_attn_record(self.model, record=flag)

    def _build_joint(self, attn, grad, device):
        joint = attn * grad if self.cfg.use_grad and grad is not None else attn
        if self.cfg.use_relu:
            joint = joint.clamp_min(0.0)
        joint = joint.mean(dim=1)
        if self.cfg.normalize in {"single", "double"}:
            joint = joint / (joint.sum(dim=-1, keepdim=True) + self.eps)
        if self.cfg.use_residual:
            identity = torch.eye(joint.shape[-1], device=device).unsqueeze(0)
            joint = joint + identity
        if self.cfg.normalize == "double":
            joint = joint / (joint.sum(dim=-1, keepdim=True) + self.eps)
        return joint

    def _rollout(self, relevance, joint):
        if relevance is None or not self.cfg.rollout:
            return joint[:, :1, :] if self.has_cls else joint.mean(dim=1, keepdim=True)
        return torch.bmm(relevance, joint)

    def _attribute_tokens(self, x, target=None, use_logits=True):
        self.model.zero_grad(set_to_none=True)
        self._enable_record_attn(True)
        try:
            logits = self.model(x)
            if target is None:
                target = logits.argmax(dim=-1)
            one_hot = torch.zeros_like(logits)
            one_hot.scatter_(1, target.reshape(-1, 1), 1.0)
            values = logits if use_logits else logits.softmax(dim=-1)
            (values * one_hot).sum().backward()

            with torch.no_grad():
                relevance = None
                blocks = self.blocks if self.cfg.rollout else self.blocks[-1:]
                for block in blocks:
                    attn_layer = getattr(block, "attn", None)
                    if not isinstance(attn_layer, LinearMultiheadAttention):
                        continue
                    attn = attn_layer.attn_map
                    if attn is None:
                        continue
                    grad = attn.grad if attn.requires_grad else None
                    joint = self._build_joint(attn, grad, x.device)
                    relevance = self._rollout(relevance, joint)
                    attn_layer.attn_map = None
                if relevance is None:
                    raise RuntimeError("No attention maps collected.")
                token_rel = relevance[:, 0, 1:] if self.has_cls else relevance[:, 0, :]
                token_rel = token_rel.clamp_min(0.0)
                return token_rel / (token_rel.amax(dim=-1, keepdim=True) + self.eps)
        finally:
            self._enable_record_attn(False)
            self.model.zero_grad(set_to_none=True)

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_rel = self._attribute_tokens(x, target, use_logits)
        return token_rel, tokens_to_heatmap(token_rel, self.model)


class TextLAGRA(LAGRA):
    def attribute(
        self,
        input_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_rel = self._attribute_tokens(input_ids, target, use_logits)
        return token_rel, token_rel.unsqueeze(1)
