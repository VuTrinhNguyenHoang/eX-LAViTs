import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

class ViTGradCAM(nn.Module):
    """
    Grad-CAM cho ViT/LinearAttention.

    Trả về:
      {'rtokens_up': [B,H,W], 'logits': [B,K], 'cam_tokens': [B,Np]}
    """
    def __init__(self, model: nn.Module,
                 target_block: int = -1,
                 hook_at: str = "attn",           # 'attn' | 'block'
                 exclude_cls: bool = True,
                 pool: str = "channel"):          # 'channel' | 'token'
        super().__init__()
        assert hook_at in {"attn", "block"}
        assert pool in {"channel", "token"}
        self.model = model
        self.block = model.blocks[target_block]
        self.mod = self.block.attn if hook_at == "attn" else self.block
        self.exclude_cls = exclude_cls and hasattr(model, "cls_token")
        self.pool = pool

        self._act = None   # [B,N,C]
        self._grad = None  # [B,N,C]

        def _fwd_hook(module, inp, out):
            self._act = out
            if isinstance(out, torch.Tensor) and out.requires_grad:
                out.register_hook(lambda g: setattr(self, "_grad", g))
        self._fh = self.mod.register_forward_hook(_fwd_hook)

        pe = self.model.patch_embed.proj
        self.stride = pe.stride[0]

    def remove_hooks(self):
        self._fh.remove()

    @staticmethod
    def _minmax01(x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        flat = x.view(B, -1)
        vmin = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        vmax = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        return (x - vmin) / (vmax - vmin + 1e-6)

    def _cam_from_AG(self, A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        """
        A,G: [B,N,C] tại module đích.
        Trả về cam_tokens: [B,Np] với/không CLS.
        """
        if self.exclude_cls:
            A = A[:, 1:, :]
            G = G[:, 1:, :]

        if self.pool == "channel":
            alpha = G.mean(dim=1, keepdim=False)                  # [B,C]
            cam = torch.relu(torch.einsum('bnc,bc->bn', A, alpha))
        else:
            cam = torch.relu((A * G).sum(dim=-1))
        return cam

    def _upsample_tokens(self, cam_tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B = cam_tokens.size(0)

        if hasattr(self.model.patch_embed, "grid_size") and self.model.patch_embed.grid_size is not None:
            Hn, Wn = self.model.patch_embed.grid_size
        else:
            Hn, Wn = H // self.stride, W // self.stride

        Ngrid = Hn * Wn
        Nt = cam_tokens.size(1)

        if Nt == Ngrid + 1 and hasattr(self.model, "cls_token"):
            cam_tokens = cam_tokens[:, 1:]
            Nt = cam_tokens.size(1)

        if Nt != Ngrid:
            raise RuntimeError(f"GradCAM: số token {Nt} không khớp lưới {Hn}x{Wn}={Ngrid}.")

        cam_map = cam_tokens.view(B, 1, Hn, Wn)
        cam_up  = F.interpolate(cam_map, size=(H, W), mode='bilinear', align_corners=False)[:, 0]
        return self._minmax01(cam_up)

    def _forward_backward_once(self, x: torch.Tensor, y_true: Optional[torch.Tensor]):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                                # [B,K]
        if y_true is None:
            y_true = logits.argmax(dim=1)
        score = logits.gather(1, y_true[:, None]).sum()
        score.backward()
        A = self._act.detach()
        G = self._grad.detach()
        return logits.detach(), A, G

    def attribute(self,
                  x: torch.Tensor,
                  y_true: Optional[torch.Tensor] = None,
                  smooth: int = 0,
                  noise_std: float = 0.15):
        assert x.dim() == 4 and x.size(1) == 3
        B, _, H, W = x.shape
        self.model.eval()
        torch.set_grad_enabled(True)
        for p in self.model.parameters():
            p.requires_grad_(True)
        
        if smooth <= 0:
            logits, A, G = self._forward_backward_once(x, y_true)
            cam_tokens = self._cam_from_AG(A, G)
            cam_up = self._upsample_tokens(cam_tokens, H, W)
            return {"rtokens_up": cam_up.detach(),
                    "logits": logits,
                    "cam_tokens": cam_tokens.detach()}

        cams = []
        last_logits = None
        x_std = x.float().flatten(1).std(dim=1).view(B, 1, 1, 1).clamp_min(1e-6)
        for _ in range(int(smooth)):
            noise = torch.randn_like(x) * (noise_std * x_std)
            logits, A, G = self._forward_backward_once(x + noise, y_true)
            last_logits = logits
            cams.append(self._cam_from_AG(A, G))
        cam_tokens = torch.stack(cams, dim=0).mean(0)
        cam_up = self._upsample_tokens(cam_tokens, H, W)
        return {"rtokens_up": cam_up.detach(),
                "logits": last_logits,
                "cam_tokens": cam_tokens.detach()}