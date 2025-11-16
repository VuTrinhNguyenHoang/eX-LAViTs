from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LARP(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        has_cls: bool = True,
        eps: float = 1e-6,
        head_agg: str = "mean",
        attn_temp: float = 0.7,
        token_p: float = 1.5,
        smooth_alpha: float = 0.3,   # tỉ lệ mix với avg-pool cho soft smoothing
    ):
        super().__init__()
        self.model = model
        self.blocks: List[nn.Module] = list(getattr(self.model, "blocks"))
        self.has_cls = has_cls
        self.eps = eps
        assert head_agg in ("mean", "sum")
        self.head_agg = head_agg

        self.attn_temp = attn_temp
        self.token_p = token_p
        self.smooth_alpha = smooth_alpha

        # grid patch (Hn, Wn) dùng cho smoothing / map heat
        self.grid_hw = getattr(getattr(self.model, "patch_embed", None), "grid_size", None)

        self._f_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: List[Dict[str, torch.Tensor]] = []

    # ----------------- utilities -----------------
    @staticmethod
    def _pos(x: torch.Tensor) -> torch.Tensor:
        return x.clamp_min(0.)

    @staticmethod
    def _lrp_linear_zplus(
        R_out: torch.Tensor,
        X: torch.Tensor,
        W: nn.Linear,
        eps: float,
    ) -> torch.Tensor:
        """
        LRP z^+ cho linear layer:
          R_in_c = sum_o ( X_c^+ W_{oc}^+ / (∑_{c'} X_{c'}^+ W_{oc'}^+ + eps) ) * R_out_o
        """
        Wp = W.weight.clamp_min(0)                 # [Cout, Cin]
        Xp = X.clamp_min(0)                        # [..., Cin]
        Z  = torch.einsum("...c,oc->...o", Xp, Wp) + eps   # [..., Cout]
        S  = R_out / Z                                     # [..., Cout]
        C  = torch.einsum("...o,oc->...c", S, Wp)          # [..., Cin]
        return Xp * C

    def _head_exists(self) -> bool:
        return hasattr(self.model, "head") and isinstance(self.model.head, nn.Linear)

    # ----------------- hook utilities -----------------
    def _clear(self):
        for h in self._f_hooks:
            h.remove()
        self._f_hooks.clear()
        self.cache.clear()

    def _register(self):
        """
        Cache (per block li):
          - x_in, x_out, attn_in, attn_out, mlp_out (detach – chỉ cần value)
          - x_grad      : grad tại input block
          - x_out_grad  : grad tại output block
          - attn_grad   : grad tại attn_out
        """
        self._clear()
        L = len(self.blocks)
        self.cache = [dict() for _ in range(L)]

        for li, blk in enumerate(self.blocks):

            # block input
            def pre_hook(li_):
                def h(module, inputs):
                    self.cache[li_]["x_in"] = inputs[0].detach()
                return h
            self._f_hooks.append(
                blk.register_forward_pre_hook(pre_hook(li))
            )

            # block output
            def post_hook(li_):
                def h(module, inputs, output):
                    self.cache[li_]["x_out"] = output.detach()
                return h
            self._f_hooks.append(
                blk.register_forward_hook(post_hook(li))
            )

            # norm1 output = attn input
            def n1_hook(li_):
                def h(module, inputs, output):
                    self.cache[li_]["attn_in"] = output.detach()
                return h
            self._f_hooks.append(
                blk.norm1.register_forward_hook(n1_hook(li))
            )

            # attn output + grad
            def attn_hook(li_):
                def h(module, inputs, output):
                    self.cache[li_]["attn_out"] = output.detach()
                    if output.requires_grad:
                        def _save_grad(g):
                            self.cache[li_]["attn_grad"] = g.detach()
                        output.register_hook(_save_grad)
                return h
            self._f_hooks.append(
                blk.attn.register_forward_hook(attn_hook(li))
            )

            # mlp output
            def mlp_hook(li_):
                def h(module, inputs, output):
                    self.cache[li_]["mlp_out"] = output.detach()
                return h
            self._f_hooks.append(
                blk.mlp.register_forward_hook(mlp_hook(li))
            )

            # backward hook: grad at x_in, x_out
            def bwd_hook(li_):
                def h(module, grad_input, grad_output):
                    if grad_input[0] is not None:
                        self.cache[li_]["x_grad"] = grad_input[0].detach()
                    if grad_output[0] is not None:
                        self.cache[li_]["x_out_grad"] = grad_output[0].detach()
                return h
            self._f_hooks.append(
                blk.register_full_backward_hook(bwd_hook(li))
            )

    # ----------------- attention weights (linear attention) -----------------
    def _attn_weights_linear(self, blk: nn.Module, attn_in: torch.Tensor) -> torch.Tensor:
        """
        Linear attention:
          s_{t,j} = <phi(q_t), phi(k_j)>
          w_{t,j} = s_{t,j} / ∑_j s_{t,j}

        Trả về: w_all [B, H, N, N]
        """
        attn = blk.attn
        B, N, C = attn_in.shape
        H = attn.h
        D = attn.d

        qkv = attn.qkv(attn_in).view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]             # [B,H,N,D]

        qf = attn._phi(q)                            # [B,H,N,D]
        kf = attn._phi(k)                            # [B,H,N,D]

        s = torch.einsum("bhnd,bhmd->bhnm", qf, kf)  # [B,H,N,N]
        s = self._pos(s) + self.eps

        if self.attn_temp != 1.0:
            s = s ** (1.0 / self.attn_temp)

        w = s / (s.sum(dim=-1, keepdim=True) + self.eps)
        return w  # [B,H,N,N]

    # ----------------- backward through one block -----------------
    def _larp_block(self, li: int, R_x2: torch.Tensor) -> torch.Tensor:
        """
        Propagate relevance qua block thứ li (channel-level).

        Input:
          R_x2: [B, N, C] – relevance tại output block (x_out).
        Output:
          R_in: [B, N, C] – relevance tại input block (x_in).
        """
        blk = self.blocks[li]
        cache = self.cache[li]

        x_in     = cache["x_in"]         # [B,N,C]
        x_out    = cache["x_out"]        # [B,N,C]
        attn_in  = cache["attn_in"]      # [B,N,C]
        attn_out = cache["attn_out"]     # [B,N,C]
        mlp_out  = cache["mlp_out"]      # [B,N,C]

        x_grad     = cache.get("x_grad", None)       # [B,N,C]
        x_out_grad = cache.get("x_out_grad", None)   # [B,N,C]
        attn_grad  = cache.get("attn_grad", None)    # [B,N,C]

        B, N, C = x_in.shape

        # ===== (1) Residual 1: x2 = x1 + MLP(x1) =====
        x1 = x_out - mlp_out  # vì x_out = x1 + mlp_out

        if x_out_grad is not None:
            # a1n = (mlp_out * x_out_grad).abs().sum(dim=-1)  # [B,N]
            a1n = (mlp_out.abs() * (x_out_grad.abs() + 0.1 * x1.abs())).sum(-1)
            b1n = (x1 * x_out_grad).abs().sum(dim=-1)
        else:
            a1n = mlp_out.abs().sum(dim=-1)
            b1n = x1.abs().sum(dim=-1)

        Z1 = a1n + b1n + self.eps
        w_mlp   = (a1n / Z1).unsqueeze(-1)   # [B,N,1]
        w_skip1 = (b1n / Z1).unsqueeze(-1)

        R_mlp_share = w_mlp * R_x2          # [B,N,C]
        R_skip1     = w_skip1 * R_x2        # [B,N,C]

        # ===== (2) MLP: LRP z^+ (fc2 -> fc1 -> x1) =====
        x1_norm = blk.norm2(x1)                          # [B,N,C]
        pre1 = blk.mlp.fc1(x1_norm)                      # [B,N,Hid]

        # q-set cho GELU/ReLU: chỉ giữ phần dương
        mask_q = (pre1 > 0).to(pre1.dtype)
        act1   = blk.mlp.act(pre1) * mask_q              # [B,N,Hid]

        R_lin2_in     = self._lrp_linear_zplus(R_mlp_share, act1, blk.mlp.fc2, self.eps)
        R_x1_from_mlp = self._lrp_linear_zplus(R_lin2_in, x1_norm, blk.mlp.fc1, self.eps)

        R_x1 = R_skip1 + R_x1_from_mlp                  # [B,N,C]

        # ===== (3) Residual 2: x1 = x_in + Attn(attn_in) =====
        if attn_grad is not None:
            a2n = (attn_out * attn_grad).abs().sum(dim=-1)    # [B,N]
        else:
            a2n = attn_out.abs().sum(dim=-1)

        if x_grad is not None:
            b2n = (x_in * x_grad).abs().sum(dim=-1)
        else:
            b2n = x_in.abs().sum(dim=-1)

        # fallback nếu đều ~0
        if (a2n + b2n).max() <= 0:
            a2n = attn_out.abs().sum(dim=-1)
            b2n = x_in.abs().sum(dim=-1)

        Z2      = a2n + b2n + self.eps
        w_attn  = (a2n / Z2).unsqueeze(-1)    # [B,N,1]
        w_skip2 = (b2n / Z2).unsqueeze(-1)

        R_attn_share = w_attn * R_x1         # [B,N,C]
        R_skip2      = w_skip2 * R_x1        # [B,N,C]

        # ===== (4) Attention mixing – gradient-weighted =====
        # token-level relevance trên nhánh attention
        R_attn_tok = R_attn_share.sum(dim=-1)          # [B,N]

        # base weights w_{t,j} từ linear attention
        w_all = self._attn_weights_linear(blk, attn_in)   # [B,H,N,N]

        if self.head_agg == "mean":
            w = w_all.mean(dim=1)                         # [B,N,N]
        else:
            w = w_all.sum(dim=1)
            w = w / (w.sum(dim=-1, keepdim=True) + self.eps)

        # gradient-aware trên source tokens j
        if x_grad is not None:
            g_src = (x_in * x_grad).abs().sum(dim=-1)                     # [B,N]
            g_src = g_src / (g_src.mean(dim=-1, keepdim=True) + self.eps)
            w = w * g_src.unsqueeze(1)                                    # scale theo j

        w = w / (w.sum(dim=-1, keepdim=True) + self.eps)                  # renorm

        # R_in_from_attn_token[j] = sum_t w[t,j] * R_attn_tok[t]
        R_in_from_attn_tok = torch.einsum("btj,bt->bj", w, R_attn_tok)    # [B,N]

        # nâng lên channel-level theo x_in^+
        x_pos    = self._pos(x_in) + self.eps                              # [B,N,C]
        chan_sum = x_pos.sum(dim=-1, keepdim=True)                         # [B,N,1]
        R_in_from_attn = x_pos / chan_sum * R_in_from_attn_tok.unsqueeze(-1)

        # ===== (5) Tổng relevance tại input block =====
        R_in = R_skip2 + R_in_from_attn                                    # [B,N,C]
        return R_in

    # ----------------- public API -----------------
    def attribute(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B,3,H,W]
        y_true: [B] (class indices)

        Trả về:
          {
            "rtokens": [B, N0],                 # relevance token-level tại input block0
            "rtokens_layers": [L+1,B,N] (opt)   # nếu cần per-layer
          }
        """
        self.model.eval()
        torch.set_grad_enabled(True)
        for p in self.model.parameters():
            p.requires_grad_(True)

        self._register()

        # ---------- forward ----------
        logits = self.model(x)                            # [B,K]
        B, K = logits.shape

        # ---------- backward để lấy grad ----------
        self.model.zero_grad(set_to_none=True)
        tgt_score = logits.gather(1, y_true[:, None]).sum()
        tgt_score.backward(retain_graph=True)

        # ---------- init relevance ở head ----------
        X_L = self.cache[-1]["x_out"]                     # [B,N_L,C]
        B2, N_L, C = X_L.shape
        assert B2 == B

        if self._head_exists() and self.has_cls:
            head: nn.Linear = self.model.head
            u = self.model.norm(X_L) if hasattr(self.model, "norm") else X_L
            u_cls = u[:, 0, :]                                            # [B,C]

            R_y = F.one_hot(y_true, num_classes=head.out_features).to(u.dtype)
            R_ucls = self._lrp_linear_zplus(R_y, u_cls, head, self.eps)   # [B,C]

            R_x2 = torch.zeros_like(X_L)                                  # [B,N,C]
            R_x2[:, 0, :] = R_ucls
        else:
            # không CLS: chia mass theo X_L^+
            Xp = self._pos(X_L) + self.eps
            mass = logits.gather(1, y_true[:, None]).squeeze(1).clamp_min(0)  # [B]
            R_x2 = Xp / Xp.sum(dim=(1, 2), keepdim=True) * mass.view(-1, 1, 1)

        # ---------- propagate ngược qua các block ----------
        R_layers_tokens: List[torch.Tensor] = [R_x2.sum(dim=-1).detach().clone()]

        for li in reversed(range(len(self.blocks))):
            R_x2 = self._larp_block(li, R_x2)
            R_layers_tokens.append(R_x2.sum(dim=-1).detach().clone())

        # token relevance tại input block0
        R_tokens = R_x2.sum(dim=-1).clamp_min(0.)                          # [B,N]

        # ---------- sharpen / focus ----------
        if self.token_p != 1.0:
            mass = R_tokens.sum(dim=1, keepdim=True) + self.eps
            R_tokens = (R_tokens ** self.token_p)
            R_tokens = R_tokens / (R_tokens.sum(dim=1, keepdim=True) + self.eps) * mass

        # ---------- soft smoothing trên lưới patch ----------
        if self.grid_hw is not None and self.smooth_alpha > 0:
            Hn, Wn = self.grid_hw
            B, N = R_tokens.shape

            if self.has_cls:
                Rt = R_tokens[:, 1:].view(B, 1, Hn, Wn)
                mass0 = Rt.sum((1, 2, 3), keepdim=True)

                smooth = F.avg_pool2d(Rt, kernel_size=3, stride=1, padding=1)
                Rt = (1.0 - self.smooth_alpha) * Rt + self.smooth_alpha * smooth
                Rt = Rt.clamp_min(0.)
                Rt = Rt * (mass0 / (Rt.sum((1, 2, 3), keepdim=True) + self.eps))

                R_tokens = torch.cat([R_tokens[:, :1], Rt.view(B, -1)], dim=1)
            else:
                Rt = R_tokens.view(B, 1, Hn, Wn)
                mass0 = Rt.sum((1, 2, 3), keepdim=True)

                smooth = F.avg_pool2d(Rt, kernel_size=3, stride=1, padding=1)
                Rt = (1.0 - self.smooth_alpha) * Rt + self.smooth_alpha * smooth
                Rt = Rt.clamp_min(0.)
                Rt = Rt * (mass0 / (Rt.sum((1, 2, 3), keepdim=True) + self.eps))

                R_tokens = Rt.view(B, -1)

        # ---------- class-conditional normalization ----------
        # đảm bảo ∑_n R_n ≈ logit f_y(x)
        scores = logits.gather(1, y_true[:, None]).squeeze(1)             # [B]
        scores = scores.clamp_min(self.eps)
        sum_tokens = R_tokens.sum(dim=1) + self.eps                       # [B]
        scale = scores / sum_tokens                                       # [B]
        R_tokens = R_tokens * scale.unsqueeze(1)

        out: Dict[str, torch.Tensor] = {"rtokens": R_tokens.detach()}

        if return_all_layers:
            R_layers_tokens = R_layers_tokens[::-1]   # [input,...,output]
            out["rtokens_layers"] = torch.stack(R_layers_tokens, dim=0).detach()

        self._clear()
        torch.set_grad_enabled(False)
        return out
