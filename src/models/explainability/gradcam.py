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
        y_true: [B] class indices
        returns: {"rtokens": [B,N]} Grad-CAM-style relevance trên tokens ở target layer
        """
        device = x.device
        self.model.eval()

        # bật grad trên model
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._register_hooks()

        # forward
        logits = self.model(x)                      # [B,K]
        B, K = logits.shape

        # target scores
        scores = logits.gather(1, y_true[:, None]).squeeze(1)  # [B]

        # backward để có grad tại target_module output
        self.model.zero_grad(set_to_none=True)
        scores.sum().backward(retain_graph=True)

        # lấy activation và gradient
        A = self._acts         # [B,N,C]
        G = self._grads        # [B,N,C]
        assert A is not None and G is not None, "Hooks chưa được kích hoạt đúng."

        # Grad-CAM token-level
        # α_c = mean_t G_{t,c}
        alpha = G.mean(dim=1)                    # [B,C]

        # h_t = ReLU( Σ_c α_c * A_{t,c} )
        # (B,N,C) * (B,1,C) → (B,N)
        h = (A * alpha.unsqueeze(1)).sum(dim=-1) # [B,N]
        h = F.relu(h)

        # chuẩn hóa mass theo scores (optional, để gần với LARP/SSRP)
        mass = h.sum(dim=1, keepdim=True) + self.eps   # [B,1]
        scores_pos = scores.clamp_min(0).unsqueeze(1)  # [B,1]
        rtokens = h * (scores_pos / mass)              # [B,N]

        self._clear_hooks()
        return {"rtokens": rtokens}