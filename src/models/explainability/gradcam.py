import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .linear_rollout import tokens_to_heatmap

class ViTGradCAM:
    """
    Grad-CAM adapted cho ViT: hook tại model.norm output [B,N,C].
    """
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        eps: float = 1e-6,
    ):
        self.model = model
        self.model.eval()
        self.has_cls = has_cls
        self.eps = eps

        self.feats: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None

        norm_module = getattr(self.model, "norm", None)
        if norm_module is None:
            raise RuntimeError("Model không có thuộc tính 'norm' để hook GradCAM.")

        def fwd_hook(module, inp, out):
            self.feats = out  # [B,N,C]

        def bwd_hook(module, grad_input, grad_output):
            # grad_output[0] tương ứng với grad wrt out
            self.grads = grad_output[0]  # [B,N,C]

        self.handle_fwd = norm_module.register_forward_hook(fwd_hook)
        self.handle_bwd = norm_module.register_full_backward_hook(bwd_hook)

    def __del__(self):
        # cleanup hook nếu object bị gc
        try:
            self.handle_fwd.remove()
            self.handle_bwd.remove()
        except Exception:
            pass

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        self.model.zero_grad(set_to_none=True)
        self.feats = None
        self.grads = None

        logits = self.model(x)  # forward; feats đã được lưu trong hook
        if target is None:
            target = logits.argmax(dim=-1)

        one_hot = torch.zeros_like(logits).to(device)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)

        if use_logits:
            out = (logits * one_hot).sum()
        else:
            probs = logits.softmax(dim=-1)
            out = (probs * one_hot).sum()

        out.backward(retain_graph=False)

        if self.feats is None or self.grads is None:
            raise RuntimeError("Không thu được features / grads trong GradCAM.")

        feats = self.feats  # [B,N,C]
        grads = self.grads  # [B,N,C]

        # alpha_k = mean grad trên tokens
        # [B,C]
        alpha = grads.mean(dim=1)

        # CAM_i = ReLU( sum_k alpha_k * feat_{i,k} )
        cam_tok = torch.einsum("bk,bnk->bn", alpha, feats)  # [B,N]
        cam_tok = cam_tok.clamp_min(0.0)

        if self.has_cls:
            patch_rel = cam_tok[:, 1:]  # bỏ CLS
        else:
            patch_rel = cam_tok

        # chuẩn hoá
        patch_rel = patch_rel / (patch_rel.amax(dim=-1, keepdim=True) + self.eps)

        heatmap = tokens_to_heatmap(patch_rel, self.model, normalize=True)
        return patch_rel, heatmap
    
class TextGradCAM:
    def __init__(self, model: nn.Module, has_cls: bool = True, eps: float = 1e-6):
        self.model = model
        self.model.eval()
        self.has_cls = has_cls
        self.eps = eps
        
        self.feats: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None

        norm_module = getattr(self.model, "norm", None)
        if norm_module is None:
            raise RuntimeError("Model không có thuộc tính 'norm' để hook GradCAM.")

        def fwd_hook(module, inp, out):
            self.feats = out  # [B,N,C]

        def bwd_hook(module, grad_input, grad_output):
            # grad_output[0] tương ứng với grad wrt out
            self.grads = grad_output[0]  # [B,N,C]

        self.handle_fwd = norm_module.register_forward_hook(fwd_hook)
        self.handle_bwd = norm_module.register_full_backward_hook(bwd_hook)

    def __del__(self):
        try:
            self.handle_fwd.remove()
            self.handle_bwd.remove()
        except Exception:
            pass
    
    def attribute(
        self,
        input_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_logits: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input_ids.device
        self.model.zero_grad(set_to_none=True)
        self.feats = None
        self.grads = None

        logits = self.model(input_ids)  # forward; feats đã được lưu trong hook
        if target is None:
            target = logits.argmax(dim=-1)

        one_hot = torch.zeros_like(logits).to(device)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)

        if use_logits:
            out = (logits * one_hot).sum()
        else:
            probs = logits.softmax(dim=-1)
            out = (probs * one_hot).sum()

        out.backward(retain_graph=False)

        if self.feats is None or self.grads is None:
            raise RuntimeError("Không thu được features / grads trong GradCAM.")

        feats = self.feats  # [B,N,C]
        grads = self.grads  # [B,N,C]

        # alpha_k = mean grad trên tokens
        # [B,C]
        alpha = grads.mean(dim=1)

        # CAM_i = ReLU( sum_k alpha_k * feat_{i,k} )
        cam_tok = torch.einsum("bk,bnk->bn", alpha, feats)  # [B,N]
        cam_tok = cam_tok.clamp_min(0.0)

        if self.has_cls:
            token_rel = cam_tok[:, 1:]  # bỏ CLS
        else:
            token_rel = cam_tok

        # chuẩn hoá
        token_rel = token_rel / (token_rel.amax(dim=-1, keepdim=True) + self.eps)

        # "heatmap" 1D: [B,1,L]
        token_rel_map = token_rel.unsqueeze(1)
        return token_rel, token_rel_map
    
