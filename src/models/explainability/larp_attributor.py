from typing import List, Dict, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch

class LARP(nn.Module):
    def __init__(self, model, has_cls: bool = True, eps: float = 1e-6, head_agg: str = "mean"):
        super().__init__()
        self.model = model
        self.blocks: List[nn.Module] = list(getattr(self.model, "blocks"))
        self.has_cls = has_cls
        self.eps = eps
        assert head_agg in ("mean", "sum")
        self.head_agg = head_agg

        self._f_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.cache: List[Dict[str, torch.Tensor]] = []

    # ----------------- hook utilities -----------------
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
            # block input
            def pre_hook(li_):
                def h(module, inputs):
                    # inputs[0]: [B, N, C]
                    self.cache[li_]["x_in"] = inputs[0].detach()
                return h
            self._f_hooks.append(
                blk.register_forward_pre_hook(pre_hook(li))
            )

            # block output
            def post_hook(li_):
                def h(module, inputs, output):
                    # output: [B, N, C]
                    self.cache[li_]["x_out"] = output.detach()
                return h
            self._f_hooks.append(
                blk.register_forward_hook(post_hook(li))
            )

            # norm1 output = attn input
            def n1_hook(li_):
                def h(module, inputs, output):
                    # output: [B, N, C]
                    self.cache[li_]["attn_in"] = output.detach()
                return h
            self._f_hooks.append(
                blk.norm1.register_forward_hook(n1_hook(li))
            )

            # attn output
            def attn_hook(li_):
                def h(module, inputs, output):
                    # output: [B, N, C]
                    self.cache[li_]["attn_out"] = output.detach()
                return h
            self._f_hooks.append(
                blk.attn.register_forward_hook(attn_hook(li))
            )

    @staticmethod
    def _pos(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.)

    def _attn_weights_linear(self, blk: nn.Module, attn_in: torch.Tensor) -> torch.Tensor:
        """
        Tính ma trận trọng số attention cho linear attention:
          s_{t,j} = <phi(q_t), phi(k_j)>
          w_{t,j} = s_{t,j} / sum_j s_{t,j}
        Trả về: w_all [B, H, N, N]
        """
        attn = blk.attn
        B, N, C = attn_in.shape
        H = attn.h
        D = attn.d

        # qkv: [B, N, 3*H*D] -> [3, B, H, N, D]
        qkv = attn.qkv(attn_in).view(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]           # [B, H, N, D]

        # feature map phi (non-negative)
        qf = attn._phi(q)                          # [B, H, N, D]
        kf = attn._phi(k)                          # [B, H, N, D]

        # s_{t,j} = <phi(q_t), phi(k_j)>
        # s: [B, H, N, N] với chỉ số (b,h,t,j)
        s = torch.einsum("bhnd,bhmd->bhnm", qf, kf)  # t = n, j = m

        s = self._pos(s) + self.eps
        w = s / (s.sum(dim=-1, keepdim=True) + self.eps)  # normalize theo j
        return w  # [B, H, N, N]

    # ----------------- backward through one block (token-level) -----------------
    def _larp_block(self, li: int, R_out_tokens: torch.Tensor) -> torch.Tensor:
        """
        Token-level LARP qua block thứ li.
        Input:
          R_out_tokens: [B, N] relevance tại output block (x_out).
        Trả về:
          R_in_tokens: [B, N] relevance tại input block (x_in).
        """

        blk = self.blocks[li]
        x_in = self.cache[li]["x_in"]       # [B, N, C]
        attn_in = self.cache[li]["attn_in"] # [B, N, C]
        attn_out = self.cache[li]["attn_out"]  # [B, N, C]

        B, N, C = x_in.shape

        # 1) Bỏ qua MLP ở mức token-level:
        #    x2 = x1 + MLP(x1), MLP là token-wise -> không trộn token.
        #    => R_x1 = R_x2 = R_out_tokens.
        R_x1 = R_out_tokens  # [B, N]

        # 2) Residual split: x1 = x_in + attn(attn_in)
        #    chia relevance R_x1 giữa skip x_in và nhánh attention theo L1-norm
        an2 = attn_out.abs().sum(dim=-1)    # [B, N]
        bn2 = x_in.abs().sum(dim=-1)        # [B, N]
        Z2 = an2 + bn2 + self.eps           # [B, N]

        R_attn_share = (an2 / Z2) * R_x1    # [B, N]
        R_skip2 = (bn2 / Z2) * R_x1         # [B, N]

        # 3) attention mixing: R_attn_share (t) -> input tokens (j) qua w_{t,j}
        w_all = self._attn_weights_linear(blk, attn_in)  # [B, H, N, N]

        # gộp các head
        if self.head_agg == "mean":
            w = w_all.mean(dim=1)           # [B, N, N]
        else:  # "sum"
            w = w_all.sum(dim=1)           # [B, N, N]
            w = w / (w.sum(dim=-1, keepdim=True) + self.eps)

        # R_in_from_attn[j] = sum_t w[t,j] * R_attn_share[t]
        # w: [B, N_t, N_j], R_attn_share: [B, N_t]
        R_in_from_attn = torch.einsum("btj,bt->bj", w, R_attn_share)  # [B, N]

        # 4) Tổng relevance tại input token của block
        R_in_tokens = R_skip2 + R_in_from_attn  # [B, N]
        return R_in_tokens

    # ----------------- public API -----------------
    @torch.no_grad()
    def attribute(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        return_all_layers: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        x: input batch với shape phù thuộc model (VD: [B, N, C] cho text, [B, 3, H, W] cho image).
           Model phải tự xử lý embed/patch.
        y_true: [B] class indices.
        return_all_layers:
          - True: trả thêm danh sách relevance per layer (từ output về input).
        
        Trả về:
          {
            "rtokens": [B, N0],  # relevance token-level tại input block0
            "rtokens_layers": Optional[List[torch.Tensor]]  # nếu return_all_layers
          }
        """

        self.model.eval()
        self._register()

        # forward
        logits = self.model(x)                 # [B, num_classes]
        B, K = logits.shape

        # lấy số token của block cuối
        X_L = self.cache[-1]["x_out"]          # [B, N_L, C]
        _, N_L, _ = X_L.shape

        # 1) khởi tạo relevance tại output token layer cuối
        #    - Nếu có CLS: gán relevance của class y_true vào CLS token.
        #    - Nếu không CLS: chia đều relevance cho tất cả token.
        scores = logits.gather(1, y_true[:, None]).squeeze(1)  # [B]
        R_tokens = torch.zeros(B, N_L, device=x.device, dtype=logits.dtype)

        if self.has_cls:
            R_tokens[:, 0] = scores
        else:
            # chia đều cho tất cả token
            R_tokens = R_tokens + scores[:, None] / float(N_L)

        # 2) lan truyền ngược qua các block (L-1 -> 0)
        R_layers: List[torch.Tensor] = [R_tokens.clone()]

        for li in reversed(range(len(self.blocks))):
            R_tokens = self._larp_block(li, R_tokens)
            R_layers.append(R_tokens.clone())

        # R_tokens hiện là relevance tại input block0
        out: Dict[str, torch.Tensor] = {"rtokens": R_tokens}

        if return_all_layers:
            # đảo ngược lại cho dễ đọc: [input, ..., output]
            R_layers = R_layers[::-1]
            out["rtokens_layers"] = torch.stack(R_layers, dim=0)  # [L+1, B, N]

        self._clear()
        return out