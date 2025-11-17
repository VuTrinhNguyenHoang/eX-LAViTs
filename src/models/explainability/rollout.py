import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class Rollout(nn.Module):
    """
    Attention Rollout: tái tạo map A từ q,k (linear attention),
    nhân chuỗi qua các block, lấy dòng CLS.

    Trả về: {'rtokens_up': [B,H,W]}.
    """
    def __init__(self, model: nn.Module,
                 start_layer: int = 0,
                 eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.blocks: List[nn.Module] = list(model.blocks)
        self.start_layer = int(start_layer)
        self.eps = float(eps)
        self.has_cls = hasattr(model, "cls_token")
        self.grid_hw = getattr(model.patch_embed, "grid_size", None)
        pe = self.model.patch_embed.proj
        self.stride = pe.stride[0]

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.attn_maps: List[torch.Tensor] = []

    def _register(self):
        self._hooks.clear()
        self.attn_maps = [None for _ in range(len(self.blocks))]

        for li, blk in enumerate(self.blocks):
            attn = blk.attn

            def make_hook(idx):
                def h(module, inputs, output):
                    x_in = inputs[0]          # [B,N,C] sau norm1
                    B, N, C = x_in.shape
                    H = getattr(module, "h", getattr(module, "num_heads", None))
                    if H is None:
                        raise RuntimeError("Không tìm được số head trong attention.")
                    D = getattr(module, "d", C // H)

                    qkv = module.qkv(x_in).view(B, N, 3, H, D).permute(2,0,3,1,4)
                    q, k = qkv[0], qkv[1]        # [B,H,N,D]

                    if hasattr(module, "_phi"):
                        qf = module._phi(q)
                        kf = module._phi(k)
                    else:
                        qf, kf = q, k

                    k_sum = kf.sum(dim=2, keepdim=True)        # [B,H,1,D]
                    den = (qf * k_sum).sum(dim=-1, keepdim=True) + self.eps  # [B,H,N,1]
                    A = torch.einsum('bhnd,bhmd->bhnm', qf, kf) / den        # [B,H,N,N]
                    A = A.mean(dim=1)                                       # [B,N,N]
                    self.attn_maps[idx] = A.detach()
                return h
            self._hooks.append(attn.register_forward_hook(make_hook(li)))

    def _clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def attribute(self,
                  x: torch.Tensor,
                  y_true: Optional[torch.Tensor] = None):
        """
        x: [B,3,H,W]
        """
        B, _, H, W = x.shape
        self.model.eval()
        with torch.no_grad():
            self._register()
            _ = self.model(x)
            self._clear()

        A_list = self.attn_maps
        assert all(a is not None for a in A_list), "Rollout: không thu được attention map."

        N = A_list[0].size(1)
        eye = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)

        joint = eye
        for A in A_list[self.start_layer:]:
            A = A.clamp_min(0)
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            A_hat = A + eye
            A_hat = A_hat / (A_hat.sum(dim=-1, keepdim=True) + self.eps)
            joint = joint @ A_hat

        if self.has_cls:
            cam_tokens = joint[:, 0, 1:]       # [B,Np]
        else:
            cam_tokens = joint.mean(dim=1)     # [B,N]

        if self.grid_hw is not None:
            Hn, Wn = self.grid_hw
        else:
            Hn, Wn = H // self.stride, W // self.stride

        cam_map = cam_tokens.view(B, 1, Hn, Wn)
        cam_up = F.interpolate(cam_map, size=(H,W), mode='bilinear', align_corners=False)[:,0]
        return {"rtokens_up": cam_up.detach()}