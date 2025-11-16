import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class ViTGradCAM(nn.Module):
    def __init__(self, model: nn.Module, has_cls: bool = True, eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.has_cls = has_cls
        self.eps = eps

        self._acts = None   # activations [B,N,C]
        self._grads = None  # gradients   [B,N,C]
        self._f_hook = None
        self._b_hook = None

        assert hasattr(self.model, "blocks") and len(self.model.blocks) > 0
        self.target_module = self.model.blocks[-1].norm2

    def _clear_hooks(self):
        if self._f_hook is not None:
            self._f_hook.remove()
            self._f_hook = None
        if self._b_hook is not None:
            self._b_hook.remove()
            self._b_hook = None
        self._acts = None
        self._grads = None

    def _register_hooks(self):
        self._clear_hooks()

        def f_hook(module, inputs, output):
            # output: [B,N,C]
            self._acts = output

        def b_hook(module, grad_input, grad_output):
            # grad_output[0]: [B,N,C]
            self._grads = grad_output[0]

        self._f_hook = self.target_module.register_forward_hook(f_hook)
        self._b_hook = self.target_module.register_full_backward_hook(b_hook)

    def attribute(self, x: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B,3,H,W]
        y_true:
            - [B] (long/int): class indices
            - hoặc [B,K]: one-hot / trọng số trên lớp
        returns: {"rtokens": [B,N]} Grad-CAM-style relevance trên tokens ở target layer
        """
        self.model.eval()

        # đảm bảo tham số model có grad
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._register_hooks()

        # luôn bật autograd, kể cả khi gọi trong no_grad
        with torch.set_grad_enabled(True):
            logits = self.model(x)          # [B,K]
            B, K = logits.shape

            # tính scores theo y_true
            if y_true.dim() == 1:
                # class indices
                if y_true.dtype != torch.long:
                    y_idx = y_true.long()
                else:
                    y_idx = y_true
                scores = logits.gather(1, y_idx.view(-1, 1)).squeeze(1)  # [B]
            else:
                # one-hot / trọng số trên lớp: [B,K]
                y_w = y_true.to(logits.dtype)
                scores = (logits * y_w).sum(dim=-1)  # [B]

            # backward để có grad tại target_module output
            self.model.zero_grad(set_to_none=True)
            scores.sum().backward(retain_graph=False)

        # lấy activation và gradient
        A = self._acts   # [B,N,C]
        G = self._grads  # [B,N,C]
        assert A is not None and G is not None, "Hooks chưa được kích hoạt đúng."

        # Grad-CAM token-level
        # α_c = mean_t G_{t,c}
        alpha = G.mean(dim=1)                       # [B,C]

        # h_t = ReLU( Σ_c α_c * A_{t,c} )
        # (B,N,C) * (B,1,C) → (B,N)
        h = (A * alpha.unsqueeze(1)).sum(dim=-1)    # [B,N]
        h = F.relu(h)

        # chuẩn hóa mass theo scores (để mass ≈ scores_pos)
        scores_pos = scores.clamp_min(0).unsqueeze(1)  # [B,1]
        mass = h.sum(dim=1, keepdim=True) + self.eps   # [B,1]
        rtokens = h * (scores_pos / mass)              # [B,N]

        self._clear_hooks()
        return {"rtokens": rtokens}
