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

@dataclass
class LAGRAConfig:
    use_grad: bool = True
    use_relu: bool = True
    use_residual: bool = True
    normalize: str = "double" # "single", "double", "none"
    rollout: bool = True

class LAGRA:
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        blocks_attr: str = "blocks",
        eps: float = 1e-6,
        config: LAGRAConfig = LAGRAConfig(),
    ):
        self.model = model
        self.model.eval()
        self.blocks: List[nn.Module] = list(getattr(model, blocks_attr))
        self.has_cls = has_cls
        self.eps = eps
        self.cfg = config

    def _enable_record_attn(self, flag: bool):
        set_linear_attn_record(self.model, record=flag)

    def _build_joint(self, attn, grad, device):
        """
        attn: [B,H,N,N]
        grad: [B,H,N,N] hoặc None
        """
        if self.cfg.use_grad and grad is not None:
            joint = attn * grad
        else:
            joint = attn
        
        if self.cfg.use_relu:
            joint = joint.clamp_min(0.0)
        
        joint = joint.mean(dim=1)  # [B,N,N]

        if self.cfg.normalize in {"single", "double"}:
            joint = joint / (joint.sum(dim=-1, keepdim=True) + self.eps)

        if self.cfg.use_residual:
            B, N, _ = joint.shape
            eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
            joint = joint + eye

        if self.cfg.normalize == "double":
            joint = joint / (joint.sum(dim=-1, keepdim=True) + self.eps)

        return joint  # [B,N,N]

    def _rollout(self, R_all, joint):
        if not self.cfg.rollout:
            return joint
        if R_all is None:
            return joint
        return torch.bmm(R_all, joint)

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
        
        score = (logits * one_hot).sum() if use_logits else \
                (logits.softmax(dim=-1) * one_hot).sum()

        self.model.zero_grad(set_to_none=True)
        score.backward()

        R_all = None
        for blk in self.blocks:
            attn_layer = getattr(blk, "attn", None)
            if not isinstance(attn_layer, LinearMultiheadAttention):
                continue
            attn = attn_layer.attn_map
            grad = attn.grad if (attn is not None and attn.requires_grad) else None
            if attn is None:
                continue

            joint = self._build_joint(attn, grad, device)
            R_all = self._rollout(R_all, joint)

        self._enable_record_attn(False)
        self.model.zero_grad(set_to_none=True)

        if R_all is None:
            raise RuntimeError("No attention maps collected.")

        if self.has_cls:
            token_rel = R_all[:, 0, 1:]
        else:
            token_rel = R_all.mean(dim=1)


        token_rel = token_rel.clamp_min(0.0)
        token_rel = token_rel / (token_rel.amax(dim=-1, keepdim=True) + self.eps)

        heatmap = tokens_to_heatmap(token_rel, self.model)
        return token_rel, heatmap
    
class TextLAGRA:
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
        input_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        self.model.zero_grad(set_to_none=True)
        self._enable_record_attn(True)

        logits = self.model(input_ids)          # [B,C]
        if target is None:
            target = logits.argmax(dim=-1)

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
            attn = attn_layer.attn_map           # [B,H,N,N]
            if attn is None:
                continue
            grad = attn_layer.attn_map.grad      # [B,H,N,N]
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
                R_all = torch.bmm(R_all, joint)   # [B,N,N]

        self._enable_record_attn(False)
        self.model.zero_grad(set_to_none=True)

        if R_all is None:
            raise RuntimeError("Không thu được attn_map nào trong TextLAGRA.")

        if self.has_cls:
            cls_idx = 0
            token_rel_full = R_all[:, cls_idx, :]  # [B,N]
            token_rel = token_rel_full[:, 1:]      # bỏ CLS → [B,L_tok]
        else:
            token_rel = R_all.mean(dim=1)          # [B,N]

        token_rel = token_rel.clamp_min(0.0)
        token_rel = token_rel / (token_rel.amax(dim=-1, keepdim=True) + self.eps)

        token_rel_map = token_rel.unsqueeze(1)     # [B,1,L_tok]
        return token_rel, token_rel_map
    
