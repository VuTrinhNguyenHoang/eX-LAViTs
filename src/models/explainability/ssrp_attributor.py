from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSRP(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 r_modes: int = 8,
                 lam: float = 0.3,
                 use_shap: bool = True,
                 shap_samples: int = 32,
                 eps: float = 1e-6):
        super().__init__()
        self.model = model
        self.r = int(r_modes)
        self.lam = float(lam)
        self.use_shap = bool(use_shap and shap_samples > 0)
        self.shap_samples = int(shap_samples)
        self.eps = float(eps)
        self.grid_hw = getattr(self.model.patch_embed, "grid_size", None)
        
        # backbone structure
        self.blocks: List[nn.Module] = list(getattr(self.model, "blocks"))
        self.has_cls = hasattr(self.model, "cls_token")
        self.embed_dim = self.blocks[0].norm1.normalized_shape[0]

        # caches
        self._f_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: List[Dict[str, torch.Tensor]] = []

        # patch embed
        self.patch_conv: nn.Conv2d = self.model.patch_embed.proj
        self.patch_size = self.patch_conv.kernel_size[0]
        self.stride = self.patch_conv.stride[0]

    # --------------------- utils ---------------------
    @staticmethod
    def _pos(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(0.)

    @staticmethod
    def _lrp_linear_zplus(R_out: torch.Tensor, X: torch.Tensor, W: nn.Linear, eps: float) -> torch.Tensor:
        # X:[B,*,Cin], W.weight:[Cout,Cin], R_out:[B,*,Cout]
        Wp = W.weight.clamp_min(0)                 # [Cout,Cin]
        Xp = X.clamp_min(0)                        # [B,*,Cin]
        Z  = torch.einsum('...c,oc->...o', Xp, Wp) + eps
        S  = R_out / Z
        C  = torch.einsum('...o,oc->...c', S, Wp)
        return Xp * C

    def _clear(self):
        for h in self._f_hooks:
            h.remove()
        self._f_hooks.clear()
        self.cache.clear()

    def _register(self):
        self._clear()
        L = len(self.blocks)
        self.cache = [dict() for _ in range(L)]

        for li, blk in enumerate(self.blocks):

            def pre_hook(li):
                def h(module, inputs):
                    self.cache[li]["x_in"] = inputs[0].detach()
                return h
            self._f_hooks.append(blk.register_forward_pre_hook(pre_hook(li)))

            def post_hook(li):
                def h(module, inputs, output):
                    self.cache[li]["x_out"] = output.detach()
                return h
            self._f_hooks.append(blk.register_forward_hook(post_hook(li)))

            def n1_hook(li):
                def h(module, inputs, output):
                    self.cache[li]["attn_in"] = output.detach()
                return h
            self._f_hooks.append(blk.norm1.register_forward_hook(n1_hook(li)))

            # capture attn_out and its grad via tensor hook
            def attn_f_hook(li):
                def h(module, inputs, output):
                    self.cache[li]["attn_out"] = output.detach()
                    if output.requires_grad:
                        def _save_grad(g): self.cache[li]["attn_grad"] = g.detach()
                        output.register_hook(_save_grad)
                return h
            self._f_hooks.append(blk.attn.register_forward_hook(attn_f_hook(li)))

            def mlp_hook(li):
                def h(module, inputs, output):
                    self.cache[li]["mlp_out"] = output.detach()
                return h
            self._f_hooks.append(blk.mlp.register_forward_hook(mlp_hook(li)))

    # --------------------- SHAP on spectral modes (targeted) ---------------------
    def _kernelshap_phi_target(self,
                               g_head: torch.Tensor,
                               qf: torch.Tensor,
                               U: torch.Tensor,
                               S: torch.Tensor,
                               W: torch.Tensor,
                               A_den: torch.Tensor,
                               Lsamp: int) -> torch.Tensor:
        """
        g_head:[B,H,N,D], qf:[B,H,N,D], U,W:[B,H,D,r], S:[B,H,r], A_den:[B,H,N,1]
        Trả về phi:[B,H,r], với y(m)=<g_head, Attn_out(m)>
        """
        B,H,N,D = qf.shape
        r = S.size(-1)
        eye = torch.eye(r, device=qf.device, dtype=qf.dtype).view(1,1,r,r)

        Ms, Ys = [], []
        for _ in range(Lsamp):
            m = torch.randint(0, 2, (B,H,r), device=qf.device, dtype=torch.int64).to(S.dtype)
            s = m.sum(-1, keepdim=True)
            # tránh mặt nạ rỗng / đầy
            m = torch.where((s==0)|(s==r), 1 - m, m)

            US = U * ((S * m).unsqueeze(2))                 # [B,H,D,r]
            KV_m = torch.matmul(US, W.transpose(-2,-1))     # [B,H,D,D]
            Attn_m = torch.einsum('bhnd,bhdd->bhnd', qf, KV_m) / (A_den + self.eps)
            y = (g_head * Attn_m).sum(dim=(2,3))            # [B,H]
            Ms.append(m); Ys.append(y)

        M = torch.stack(Ms, dim=2)                          # [B,H,L,r]
        Y = torch.stack(Ys, dim=2)                          # [B,H,L]

        s = M.float().mean(-1, keepdim=True).clamp_(1e-6, 1-1e-6)
        w = (r - 1) / (s * (1 - s))                         # kernel Shapley

        MtW = (M * w).transpose(-2, -1)                     # [B,H,r,L]
        A = MtW @ M + 1e-6 * eye                            # [B,H,r,r]
        b = (MtW @ Y.unsqueeze(-1))                         # [B,H,r,1]
        phi = torch.linalg.solve(A, b).squeeze(-1)          # [B,H,r]

        # enforce efficiency: sum_r phi_r = y(1) - y(0)=y(1), với y(0)=0
        US_full = U * (S.unsqueeze(2))                      # [B,H,D,r]
        KV_full = torch.matmul(US_full, W.transpose(-2,-1)) # [B,H,D,D]
        Attn_full = torch.einsum('bhnd,bhdd->bhnd', qf, KV_full) / (A_den + self.eps)
        y1 = (g_head * Attn_full).sum(dim=(2,3))            # [B,H]
        phi = phi * ((y1.unsqueeze(-1)) / (phi.sum(-1, keepdim=True) + 1e-6))
        return phi
    
    # --------------------- one block SSRP ---------------------
    def _ssrp_block(self, li: int, R_x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:  R_x1 at x1 (sau residual attn), [B,N,C]
        Output: R_skip (về skip attn), R_attn_tokensC (N×C)
        """
        blk = self.blocks[li]
        attn = blk.attn
        B,N,C = R_x1.shape
        H = attn.h; D = attn.d

        x_in     = self.cache[li]["x_in"]            # [B,N,C]
        attn_in  = self.cache[li]["attn_in"]         # [B,N,C]
        attn_out = self.cache[li]["attn_out"]        # [B,N,C]
        g_attn   = self.cache[li]["attn_grad"]       # [B,N,C]

        # residual split 2: x1 = x_in + attn(attn_in)
        a2 = attn_out; b2 = x_in
        a2n = blk.norm1(a2).clamp_min(0).abs().sum(-1, keepdim=True)
        b2n = blk.norm1(b2).clamp_min(0).abs().sum(-1, keepdim=True)
        Z2  = a2n + b2n + self.eps
        R_attn_share = (a2n / Z2) * R_x1
        R_skip2      = (b2n / Z2) * R_x1

        mass_attn = R_attn_share.sum(dim=(1,2), keepdim=True)   # [B,1,1]

        # q,k,v (như forward)
        qkv = attn.qkv(attn_in).view(B, N, 3, H, D).permute(2,0,3,1,4)  # [3,B,H,N,D]
        q, k, v = qkv[0], qkv[1], qkv[2]                                # [B,H,N,D]
        qf = attn._phi(q)
        kf = attn._phi(k)

        # KV và SVD top-r
        kv = torch.matmul(kf.transpose(-2, -1), v)             # [B,H,D,D]
        try:
            U, S, Vh = torch.linalg.svd(kv, full_matrices=False)
        except RuntimeError:
            # fallback pca_lowrank
            U, S, V = torch.pca_lowrank(kv.reshape(B*H, D, D), q=min(self.r, D))
            U = U.view(B,H,D,-1); S = S.view(B,H,-1); Vh = V.view(B,H,D,-1).transpose(-2,-1)
        r = min(self.r, D)
        U = U[..., :r]                                         # [B,H,D,r]
        S = S[..., :r]                                         # [B,H,r]
        W = Vh.transpose(-2, -1)[..., :r]                      # [B,H,D,r]

        # token-mode projection và gradient theo head
        z  = torch.einsum('bhnd,bhdr->bhnr', qf, U)            # [B,H,N,r]
        Wo = attn.out_proj.weight                              # [C,C]
        g_head = torch.einsum('bnc,cm->bnm', g_attn, Wo.t()).view(B,N,H,D).permute(0,2,1,3).contiguous()  # [B,H,N,D]
        Pi = torch.einsum('bhdr,bhnd->bhnr', W, g_head)        # [B,H,N,r]

        # y_den của linear attention
        z_kf  = kf.sum(dim=2)                                  # [B,H,D]
        A_den = torch.einsum('bhnd,bhd->bhn', qf, z_kf).unsqueeze(-1).clamp_min(self.eps)  # [B,H,N,1]

        # head weights theo năng lượng gradient (mạnh theo lớp mục tiêu)
        alpha_raw = (g_head.pow(2).sum(dim=(2,3)) * (S.sum(-1) + self.eps))  # [B,H]
        alpha = alpha_raw / (alpha_raw.sum(dim=1, keepdim=True) + self.eps)  # [B,H]
        R_head = (alpha.unsqueeze(-1) * mass_attn).squeeze(-1)               # [B,H]

        # scores theo mode
        g_mode = S * (z * Pi).sum(dim=2)                        # [B,H,r]
        w_mode = (g_mode.abs() + self.eps)
        w_mode = w_mode / (w_mode.sum(dim=-1, keepdim=True) + self.eps)      # [B,H,r]
        R_spec = R_head.unsqueeze(-1) * w_mode                                  # [B,H,r]

        # SHAP theo mục tiêu
        if self.use_shap and self.lam > 0:
            Lsamp = max(self.shap_samples, 2*r)
            phi = self._kernelshap_phi_target(g_head, qf, U, S, W, A_den, Lsamp)   # [B,H,r]
            phi_pos  = phi.clamp_min(0)
            phi_norm = phi_pos / (phi_pos.sum(dim=-1, keepdim=True) + self.eps)    # [B,H,r]
            R_shap   = R_head.unsqueeze(-1) * phi_norm                              # [B,H,r]
        else:
            R_shap = torch.zeros_like(R_spec)

        # phân bổ mode→token: A_{n,r} ∝ z^+ * Pi^+ ; chuẩn hoá theo n cho từng r
        A = (z.clamp_min(0) * Pi.clamp_min(0))                   # [B,H,N,r]
        A = A / (A.sum(dim=2, keepdim=True) + self.eps)          # \sum_n A = 1

        R_tokens_LRP  = (A * R_spec.unsqueeze(2)).sum(dim=-1)    # [B,H,N]
        R_tokens_SHAP = (A * R_shap.unsqueeze(2)).sum(dim=-1)    # [B,H,N]
        R_tokens = ((1 - self.lam) * R_tokens_LRP + self.lam * R_tokens_SHAP).sum(dim=1)  # [B,N]
        R_tokens = R_tokens.clamp_min(0)
        # conservation: \sum_n R_tokens = \sum_h R_head = mass_attn

        if self.grid_hw is not None:
            Hn, Wn = self.grid_hw
        else:
            Np = (N - 1) if self.has_cls else N
            Hn = int(math.sqrt(Np)); Wn = max(1, Np // Hn)

        Rt_all = R_tokens  # [B,N]
        if self.has_cls:
            Rt = (R_tokens[:, 1:] if self.has_cls else R_tokens).view(B,1,Hn,Wn)
            mass0 = Rt.sum((1,2,3), keepdim=True)
            Rt = F.avg_pool2d(Rt, kernel_size=5, stride=1, padding=2)
            Rt = Rt * (mass0 / (Rt.sum((1,2,3), keepdim=True) + self.eps))
            R_tokens = torch.cat([R_tokens[:, :1], Rt.view(B, -1)], dim=1) if self.has_cls else Rt.view(B,-1)
        else:
            Rt = Rt_all.view(B, 1, Hn, Wn)
            mass0 = Rt.view(B, -1).sum(1, keepdim=True)
            Rt = F.avg_pool2d(Rt, kernel_size=5, stride=1, padding=1)
            mass1 = Rt.view(B, -1).sum(1, keepdim=True) + self.eps
            R_tokens = Rt.view(B, -1) * (mass0 / mass1)
        
        # nâng R_tokens → N×C theo tỉ lệ kênh dương của x_in
        x_pos = self._pos(x_in) + self.eps
        chan_sum = x_pos.sum(dim=-1, keepdim=True)               # [B,N,1]
        R_attn_tokensC = x_pos / chan_sum * R_tokens.unsqueeze(-1)
        return R_skip2, R_attn_tokensC

    # --------------------- pixel projection ---------------------
    def _tokens_to_pixels(self, R_tokensC: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B, Cin, H, W = x.shape
            P, S = self.patch_size, self.stride
            conv: nn.Conv2d = self.patch_conv
            D = conv.out_channels
    
            R_patch = R_tokensC[:, 1:, :] if self.has_cls else R_tokensC  # [B,Np,D]
    
            # 1) ảnh dương ổn định
            x_pos = x - x.amin(dim=(2,3), keepdim=True)  # ≥0
    
            # 2) unfold
            patches = F.unfold(x_pos, kernel_size=P, stride=S)     # [B,K,Np]
            K = patches.size(1)
            Wpos = self._pos(conv.weight).view(D, K)               # [D,K]
    
            # 3) cửa sổ: chỉ dùng Hann khi có chồng lấn
            if S < P:
                w1 = torch.hann_window(P, device=x.device, dtype=x.dtype)
                w2 = (w1[:, None] * w1[None, :])                   # [P,P]
                w2 = (w2 / w2.mean()).clamp_min(1e-6)
                wflat = w2.reshape(1, 1, P*P).repeat(1, Cin, 1).reshape(1, Cin*P*P, 1)  # [1,K,1]
            else:
                wflat = torch.ones(1, Cin*P*P, 1, device=x.device, dtype=x.dtype)
    
            patches_w = patches * wflat
    
            # 4) mẫu số và phân bổ trong patch
            bpos = 0.0
            if self.patch_conv.bias is not None:
                bpos = self.patch_conv.bias.clamp_min(0).view(1, D, 1)
            denom = torch.einsum('bkn,dk->bdn', patches_w, Wpos) + bpos + self.eps  # [B,D,Np]
            T = (R_patch.permute(0, 2, 1)) / denom                                  # [B,D,Np]
            Smap = torch.einsum('bdn,dk->bkn', T, Wpos)                              # [B,K,Np]
            contrib = patches_w * Smap                                              # [B,K,Np]
    
            # 5) fold + bù chồng lấn
            overlap = F.fold(wflat.expand(B, -1, R_patch.size(1)), (H, W),
                             kernel_size=P, stride=S)                                # [B,1,H,W]
            overlap = overlap.expand(B, Cin, H, W)
            Rpix = F.fold(contrib, (H, W), kernel_size=P, stride=S) / (overlap + 1e-6)
    
            # 6) chặn âm và làm mượt bảo toàn khối lượng (khử viền)
            Rpix = Rpix.clamp_min(0)
            ksz, sigma = 5, 1.0
            ax = torch.arange(ksz, device=x.device, dtype=x.dtype) - (ksz-1)/2
            g1 = torch.exp(-(ax**2)/(2*sigma**2)); g2 = (g1[:,None]*g1[None,:]); g2 /= g2.sum()
            kernel = g2.view(1,1,ksz,ksz).repeat(Cin,1,1,1)
            Rsm = F.conv2d(Rpix, kernel, padding=ksz//2, groups=Cin)
            Rpix = Rsm * (Rpix.sum((2,3), keepdim=True) / (Rsm.sum((2,3), keepdim=True) + 1e-6))
    
            return Rpix

    # --------------------- main API ---------------------
    @torch.no_grad()
    def _head_exists(self) -> bool:
        return hasattr(self.model, "head") and isinstance(self.model.head, nn.Linear)

    def attribute(self, x: torch.Tensor, y_true: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x:[B,3,H,W], y_true:[B] class indices
        returns: {'rpix':[B,3,H,W], 'rtokens_up':[B,H,W]}
        """
        self.model.eval()
        torch.set_grad_enabled(True)
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._register()
        logits = self.model(x)                                  # [B,K]

        # backprop để có grad tại attn_out
        self.model.zero_grad(set_to_none=True)
        tgt_score = logits.gather(1, y_true[:, None]).sum()
        tgt_score.backward(retain_graph=True)

        # init R ở head bằng z+-ε
        X_L = self.cache[-1]["x_out"]                           # [B,N,C]
        if self._head_exists() and self.has_cls:
            head: nn.Linear = self.model.head
            u = self.model.norm(X_L) if hasattr(self.model, "norm") else X_L
            u_cls = u[:, 0, :]                                  # [B,C]
            R_y = F.one_hot(y_true, num_classes=head.out_features).to(u.dtype)  # [B,K]
            R_ucls = self._lrp_linear_zplus(R_y, u_cls, head, self.eps)         # [B,C]
            R_x2 = torch.zeros_like(X_L); R_x2[:, 0, :] = R_ucls
        else:
            # không có CLS: phân bổ theo X_L^+
            Xp = self._pos(X_L) + self.eps
            mass = logits.gather(1, y_true[:,None]).squeeze(1).clamp_min(0)
            R_x2 = Xp / Xp.sum(dim=(1,2), keepdim=True) * mass.view(-1,1,1)

        # duyệt ngược các block
        for li in reversed(range(len(self.blocks))):
            blk = self.blocks[li]
            x_in    = self.cache[li]["x_in"]
            attn_out= self.cache[li]["attn_out"]
            x1 = x_in + attn_out

            # residual split 1: x2 = x1 + mlp(x1)
            a1 = self.cache[li]["mlp_out"]; b1 = x1
            a1n = blk.norm2(a1).clamp_min(0).abs().sum(-1, keepdim=True)
            b1n = blk.norm2(b1).clamp_min(0).abs().sum(-1, keepdim=True)
            Z1  = a1n + b1n + self.eps
            R_mlp_share = (a1n / Z1) * R_x2
            R_skip1     = (b1n / Z1) * R_x2

            # MLP: z+ với q-set cho GELU
            x1_norm = blk.norm2(x1)
            pre1 = blk.mlp.fc1(x1_norm)                          # [B,N,Hid]
            mask_q = (pre1 > 0).to(pre1.dtype)                    # q-set approx
            act1 = blk.mlp.act(pre1) * mask_q

            R_lin2_in   = self._lrp_linear_zplus(R_mlp_share, act1, blk.mlp.fc2, self.eps)
            R_x1_from_mlp = self._lrp_linear_zplus(R_lin2_in, x1_norm, blk.mlp.fc1, self.eps)
            R_x1 = R_skip1 + R_x1_from_mlp

            # Attention SSRP
            R_skip2, R_attn_tokensC = self._ssrp_block(li, R_x1)
            R_x2 = R_skip2 + R_attn_tokensC

        # tokens tại input block 0
        R_tokensC0 = R_x2
        Rpix = self._tokens_to_pixels(R_tokensC0, x)             # [B,3,H,W]

        # upsample token map (không bắt buộc)
        R_tokens = R_tokensC0.sum(dim=-1)                        # [B,N]
        Hn, Wn = x.shape[-2] // self.stride, x.shape[-1] // self.stride
        R_map = R_tokens[:, 1:] if self.has_cls else R_tokens
        R_map = R_map.view(x.size(0), 1, Hn, Wn)
        R_up  = F.interpolate(R_map, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]

        self._clear()
        return {"rpix": Rpix.detach(), "rtokens_up": R_up.detach()}