import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class KernelSHAP(nn.Module):
    """
    KernelSHAP xấp xỉ trên không gian patch.

    Hiện tại triển khai đơn giản với batch=1 (thường dùng cho XAI).
    Trả về: {'rtokens_up': [1,H,W]}.
    """
    def __init__(self, model: nn.Module,
                 nsamples: int = 256):
        super().__init__()
        self.model = model
        self.nsamples = int(nsamples)
        pe = self.model.patch_embed.proj
        self.patch_size = pe.kernel_size[0]
        self.stride = pe.stride[0]

    def _grid_hw(self, x: torch.Tensor):
        if hasattr(self.model.patch_embed, "grid_size") and self.model.patch_embed.grid_size is not None:
            return self.model.patch_embed.grid_size
        B, _, H, W = x.shape
        return H // self.patch_size, W // self.patch_size

    def attribute(self,
                  x: torch.Tensor,
                  y_true: Optional[torch.Tensor] = None,
                  baseline: Optional[torch.Tensor] = None,
                  nsamples: Optional[int] = None):
        assert x.size(0) == 1, "KernelSHAPAttributor hiện chỉ hỗ trợ batch=1."
        self.model.eval()
        with torch.no_grad():
            if y_true is None:
                logits = self.model(x)
                y_true = logits.argmax(dim=1)

        if baseline is None:
            baseline = torch.zeros_like(x)

        nsamples = int(nsamples or self.nsamples)
        device = x.device
        B, _, H, W = x.shape
        Hn, Wn = self._grid_hw(x)
        Np = Hn * Wn

        delta = x - baseline

        Ms = []
        Ys = []
        for _ in range(nsamples):
            m = torch.randint(0, 2, (Np,), device=device, dtype=torch.float32)
            s = m.sum()
            if s == 0 or s == Np:
                m = 1.0 - m
            Ms.append(m)

            mask = m.view(1,1,Hn,Wn)
            mask_up = F.interpolate(mask, size=(H,W), mode='nearest')
            x_m = baseline + mask_up * delta

            with torch.no_grad():
                logits_m = self.model(x_m)
                y_m = logits_m.gather(1, y_true[:,None]).squeeze(1)
            Ys.append(y_m)

        M = torch.stack(Ms, dim=0)          # [L,Np]
        Y = torch.stack(Ys, dim=0).view(-1)          # [L]

        s_frac = M.mean(dim=1, keepdim=True).clamp_(1e-6, 1-1e-6)
        w = (Np - 1) / (s_frac * (1 - s_frac))      # [L,1]
        MtW = (M * w).t()                           # [Np,L]
        A = MtW @ M + 1e-6 * torch.eye(Np, device=device)
        b = MtW @ Y.unsqueeze(-1)                   # [Np,1]
        phi = torch.linalg.solve(A, b).squeeze(-1)  # [Np]

        phi_pos = phi.clamp_min(0)
        if phi_pos.max() > 0:
            phi_pos = phi_pos / phi_pos.max()

        cam = phi_pos.view(1,1,Hn,Wn)
        cam_up = F.interpolate(cam, size=(H,W), mode='bilinear', align_corners=False)[:,0]
        return {"rtokens_up": cam_up.detach()}