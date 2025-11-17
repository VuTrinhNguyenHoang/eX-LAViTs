import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Occlusion(nn.Module):
    """
    Occlusion trên từng patch: che 1 patch bằng baseline rồi đo độ giảm logit.
    Trả về: {'rtokens_up': [1,H,W]}.
    """
    def __init__(self, model: nn.Module,
                 baseline_value: float = 0.0):
        super().__init__()
        self.model = model
        self.baseline_value = float(baseline_value)
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
                  baseline: Optional[torch.Tensor] = None):
        assert x.size(0) == 1, "OcclusionAttributor hiện chỉ hỗ trợ batch=1."
        self.model.eval()
        device = x.device
        B, C, H, W = x.shape
        Hn, Wn = self._grid_hw(x)
        Np = Hn * Wn

        if baseline is None:
            baseline = torch.zeros_like(x) + self.baseline_value

        with torch.no_grad():
            if y_true is None:
                logits = self.model(x)
                y_true = logits.argmax(dim=1)
            logits0 = self.model(x)
            score0 = logits0.gather(1, y_true[:,None]).squeeze(1)  # [1]

        delta = x - baseline
        scores = []

        for i in range(Np):
            mask = torch.ones(1,1,Hn,Wn, device=device)
            mask.view(-1)[i] = 0.0
            mask_up = F.interpolate(mask, size=(H,W), mode='nearest')
            x_occ = baseline + mask_up * delta

            with torch.no_grad():
                logits = self.model(x_occ)
                score = logits.gather(1, y_true[:,None]).squeeze(1)
            diff = score0 - score
            scores.append(diff)

        s = torch.stack(scores, dim=0).view(Np)  # [Np]
        s_pos = s.clamp_min(0)
        if s_pos.max() > 0:
            s_pos = s_pos / s_pos.max()

        cam = s_pos.view(1,1,Hn,Wn)
        cam_up = F.interpolate(cam, size=(H,W), mode='bilinear', align_corners=False)[:,0]
        return {"rtokens_up": cam_up.detach()}