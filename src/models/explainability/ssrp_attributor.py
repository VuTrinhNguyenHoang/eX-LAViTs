from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def _phi(x: torch.Tensor, kernel: str) -> torch.Tensor:
    if kernel == "elu":
        return F.elu(x, 1.0) + 1.0
    elif kernel == "relu":
        return F.relu(x)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

def _topr_svd(B: torch.Tensor, r: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B_ = B
    B, H, D, Dv = B_.shape
    # torch.linalg.svd là svd dày; ta cắt top-r
    U, S, Vh = torch.linalg.svd(B_, full_matrices=False)  # U:[B,H,D,min], S:[B,H,min], Vh:[B,H,min,Dv]
    r = min(r, S.shape[-1])
    U = U[..., :r]                 # [B,H,D,r]
    S = S[..., :r]                 # [B,H,r]
    W = Vh[..., :r, :].transpose(-2, -1)  # [B,H,r,Dv]  (W = V)
    return U, S, W

def _kernelshap_weights(m: torch.Tensor) -> torch.Tensor:
    M, r = m.shape
    s = m.sum(dim=1)
    w = torch.zeros(M, device=m.device, dtype=m.dtype)
    valid = (s > 0) & (s < r)
    w[valid] = (r - 1) / (s[valid] * (r - s[valid]))
    return w

def _weighted_linreg(Mm: torch.Tensor, y: torch.Tensor, w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Giải beta trong y ≈ Mm @ beta bằng bình phương tối thiểu có trọng số.
    Mm: [M, r], y: [M], w: [M] (KernelSHAP weight), beta: [r]
    """
    # Nếu mẫu ít, dùng pseudo-inverse ổn định
    if Mm.shape[0] < Mm.shape[1] + 2:
        return torch.linalg.pinv(Mm) @ y

    W = torch.diag(w)  # [M,M]
    A = Mm.T @ W @ Mm  # [r,r]
    b = Mm.T @ W @ y   # [r]
    # regularize nhẹ
    A = A + eps * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    beta = torch.linalg.solve(A, b)
    return beta

def _make_random_masks(r: int, M: int, device: torch.device) -> torch.Tensor:
    """
    Tạo M mặt nạ nhị phân trên r người chơi, loại s=0 và s=r.
    Phân phối s ~ Uniform{1..r-1}.
    """
    M = max(M, 2 * r)
    s = torch.randint(low=1, high=r, size=(M,), device=device)
    masks = torch.zeros(M, r, device=device)
    for i in range(M):
        idx = torch.randperm(r, device=device)[: s[i]]
        masks[i, idx] = 1.0
    return masks

@dataclass
class SSRPConfig:
    target_layer_names: List[str]                 # tên module attention theo model.named_modules()
    r: int = 16                                   # số mode phổ top-r
    lambda_: float = 0.5                          # trộn LRP vs Shapley
    eps: float = 1e-6                             # ổn định chia
    n_shapley_samples: int = 256                  # số mẫu KernelSHAP
    kernel: str = "elu"                           # "elu" hoặc "relu" (khớp attention)
    head_weighting: str = "energy"                # "energy" hoặc "uniform"
    positivity: bool = True                       # kẹp âm về 0 rồi renormalize
    use_grad_backproj: bool = True                # lấy w_bar bằng grad tại output attention
    device: Optional[torch.device] = None         # auto nếu None

class SSRPAttributor(nn.Module):
    def __init__(self, model: nn.Module, cfg: SSRPConfig):
        self.model = model.eval()
        self.cfg = cfg
        self.device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.caches: Dict[str, Dict[str, torch.Tensor]] = {}
        self.outs: Dict[str, torch.Tensor] = {}
        self.grads: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()

    # -------------- Hooks --------------
    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if any(tn == name or tn in name for tn in self.cfg.target_layer_names):
                # forward: cache Q,K,V và output Y
                self.hooks.append(module.register_forward_hook(self._fwd_hook(name)))
                # backward: grad w.r.t. Y
                self.hooks.append(module.register_full_backward_hook(self._bwd_hook(name)))

    def _fwd_hook(self, lname: str):
        def fn(mod, inps, out):
            x_in = inps[0]             # [B,N,C] trước attention (đã norm)
            B, N, C = x_in.shape
            H, D = mod.h, mod.d

            # Lấy Q,K,V ngay trên module để nhất quán với forward
            qkv = mod.qkv(x_in).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)  # [3,B,H,N,D]
            q, k, v = qkv[0].detach(), qkv[1].detach(), qkv[2].detach()

            self.caches[lname] = {"q": q, "k": k, "v": v, "H": H, "D": D, "N": N}
            self.outs[lname] = out  # [B,N,C], sau out_proj(proj_drop)
        return fn
    
    def _bwd_hook(self, lname: str):
        def fn(mod, grad_in, grad_out):
            # grad_out[0] là grad w.r.t. output tensor của module
            self.grads[lname] = grad_out[0].detach()
        return fn
    
    def remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def _clear_buffers(self):
        self.caches.clear()
        self.outs.clear()
        self.grads.clear()

    @torch.no_grad()
    def _head_weights(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: [B,H,r] singular values
        Trả: [B,H] trọng số head chuẩn hoá
        """
        B, H, _ = S.shape
        if self.cfg.head_weighting == "energy":
            w = S.sum(dim=-1)  # [B,H]
            w = w / (w.sum(dim=1, keepdim=True) + self.cfg.eps)
        else:
            w = torch.full((B, H), 1.0 / H, device=S.device, dtype=S.dtype)
        return w  # [B,H]
    
    def _kernelshap(self, g: torch.Tensor, M: int) -> torch.Tensor:
        """
        g: [B,H,r] hệ số tuyến tính thật (surrogate), v(S)=masks @ g
        Trả: phi_spec ≈ g (KernelSHAP regression)
        """
        B, H, r = g.shape
        device = g.device
        masks = _make_random_masks(r, M, device)         # [M,r]
        weights = _kernelshap_weights(masks)             # [M]

        phi = torch.zeros_like(g)
        for b in range(B):
            for h in range(H):
                y = masks @ g[b, h]                      # [M]
                beta = _weighted_linreg(masks, y, weights, eps=self.cfg.eps)  # [r]
                phi[b, h] = beta
        return phi  # [B,H,r]
    
    def attribute(self, x: torch.Tensor, target: int) -> torch.Tensor:
        """
        Trả: heatmap [B, N] với sum_i R_i = F_t(x).
        """
        self._clear_buffers()
        x = x.to(self.device)

        # tắt grad tham số, chỉ lấy grad output attention
        req_flags = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        # 1) Forward + backward 1 lần
        logits = self.model(x)                          # [B,C]
        B, C = logits.shape
        assert 0 <= target < C
        Ft = logits[:, target]                          # [B]
        Ft_sum = Ft.sum()
        self.model.zero_grad(set_to_none=True)
        Ft_sum.backward(retain_graph=True)

        # 2) Cho từng layer attention đích, tính relevance
        relev_layers: List[torch.Tensor] = []

        for lname in self.cfg.target_layer_names:
            cache = self.caches[lname]
            q = cache["q"].to(self.device)  # [B,H,N,D]
            k = cache["k"].to(self.device)
            v = cache["v"].to(self.device)
            H, N, D = cache["H"], cache["N"], cache["D"]

            # Grad tại output attention module
            y_grad = self.grads[lname].to(self.device)   # [B,N,C]
            # Xấp xỉ chia C đều cho H để reshape: [B,H,N,D]
            try:
                y_grad = y_grad.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
            except RuntimeError:
                # fallback: suy luận H,D từ module (ít khi gặp nếu model đúng cấu trúc)
                y_grad = y_grad.reshape(B, N, H, -1).permute(0, 2, 1, 3).contiguous()
            
            w_bar = y_grad  # surrogate cục bộ theo token/head/value-dim

            # Kernel features
            qf = _phi(q, self.cfg.kernel)                # [B,H,N,D]
            kf = _phi(k, self.cfg.kernel)                # [B,H,N,D]

            # B = Phi(K)^T V
            kv = torch.matmul(kf.transpose(-1, -2).float(), v.float()).to(v.dtype)  # [B,H,D,Dv], ở đây Dv=D
            U, S, W = _topr_svd(kv, self.cfg.r)         # [B,H,D,r], [B,H,r], [B,H,r,Dv]

            # z = U^T Phi(Q): [B,H,N,r]
            z = torch.einsum("bhnd,bhdr->bhnr", qf, U)

            # proj = <W[:,j], w_bar_i> theo token i: [B,H,N,r]
            proj = torch.einsum("bhrd,bhnd->bhnr", W, w_bar)

            # Hệ số tuyến tính theo mode: g_j = sigma_j * sum_i z_{i,j} * proj_{i,j}
            g = torch.einsum("bhr,bhnr,bhnr->bhr", S, z, proj)  # [B,H,r]

            # ===== LRP-spec =====
            num = g.abs() + self.cfg.eps                        # [B,H,r]
            den = num.sum(dim=-1, keepdim=True)                 # [B,H,1]
            R_spec = (num / den) * Ft.view(B, 1, 1)             # [B,H,r]

            z_pos = z.clamp_min(0.0)
            denom_patch = z_pos.sum(dim=2, keepdim=True) + self.cfg.eps  # [B,H,1,r]
            R_lrp_h = (z_pos / denom_patch) * R_spec.unsqueeze(2)        # [B,H,N,r]
            R_lrp_h = R_lrp_h.sum(dim=-1)                                 # [B,H,N]
        
            # ===== Shapley-spec (KernelSHAP) =====
            phi_spec = self._kernelshap(g, self.cfg.n_shapley_samples)    # [B,H,r]
            R_shap_h = (z_pos / denom_patch) * phi_spec.unsqueeze(2)      # [B,H,N,r]
            R_shap_h = R_shap_h.sum(dim=-1)                                # [B,H,N]

            # ===== Head weighting + trộn lambda =====
            head_w = self._head_weights(S)                                 # [B,H]
            head_w_ = head_w.unsqueeze(-1)                                  # [B,H,1]

            R_lrp = (R_lrp_h * head_w_).sum(dim=1)                         # [B,N]
            R_shap = (R_shap_h * head_w_).sum(dim=1)                       # [B,N]
            R_l = (1.0 - self.cfg.lambda_) * R_lrp + self.cfg.lambda_ * R_shap  # [B,N]

            if self.cfg.positivity:
                R_l = R_l.clamp_min(0.0)
            # Bảo toàn cục bộ: sum_i R_l = F_t(x)
            scale_l = (Ft / (R_l.sum(dim=1) + self.cfg.eps)).view(B, 1)
            R_l = R_l * scale_l

            relev_layers.append(R_l)

        # 3) Tổng hợp nhiều layer: trung bình, positivity, renorm cuối
        R_all = torch.stack(relev_layers, dim=0).mean(dim=0)  # [B,N]
        if self.cfg.positivity:
            R_all = R_all.clamp_min(0.0)
        scale = (Ft / (R_all.sum(dim=1) + self.cfg.eps)).view(B, 1)
        R_all = R_all * scale  # conservation cuối

        # khôi phục requires_grad ban đầu
        for (p, flag) in zip(self.model.parameters(), req_flags):
            p.requires_grad_(flag)

        return R_all  # [B,N]
