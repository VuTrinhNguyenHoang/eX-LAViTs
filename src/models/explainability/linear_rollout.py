from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class LARollout(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 has_cls: bool = True, 
                 use_abs_grad: bool = True,
                 clamp_grad_pos: bool = True,
                 eps: float = 1e-6
                 ):
        super().__init__()
        self.model = model
        self.blocks: List[nn.Module] = list(getattr(model, "blocks"))
        self.has_cls = has_cls
        self.eps = eps

        self.use_abs_grad = use_abs_grad
        self.clamp_grad_pos = clamp_grad_pos

        self.grid_hw: Optional[Tuple[int, int]] = getattr(
            getattr(model, "patch_embed", None), "grid_size", None
        )

    def _tokens_to_heatmap(
        self,
        token_scores: torch.Tensor,
        img_size: Optional[Tuple[int, int]] = None
    ):
        B, num_patches = token_scores.shape
        if self.grid_hw is not None:
            H_p, W_p = self.grid_hw
        else:
            side = int(num_patches ** 0.5)
            H_p, W_p = side, side

        assert H_p * W_p == num_patches, "Số patch không khớp grid_size."
        
        maps = token_scores.reshape(B, 1, H_p, W_p)
        maps = maps - maps.amin(dim=(2, 3), keepdim=True)
        maps = maps / (maps.amax(dim=(2, 3), keepdim=True) + self.eps)

        if img_size is not None:
            H_img, W_img = img_size
        else:
            H_img, W_img = H_p * 16, W_p * 16

        maps_up = F.interpolate(
            maps, size=(H_img, W_img), mode="bilinear", align_corners=False
        )
        return maps_up
    
    def _collect_attn_and_grads(self):
        attn_maps = []
        attn_grads = []

        for blk in self.blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is None:
                raise RuntimeError("Block không có thuộc tính 'attn'.")

            attn = getattr(attn_mod, "attn_map", None)
            if attn is None:
                raise RuntimeError(
                    "block.attn.attn_map is None. "
                    "Hãy đảm bảo LinearMultiheadAttention lưu self.attn_map trong forward."
                )

            if attn.grad is None:
                raise RuntimeError(
                    "attn_map.grad is None. "
                    "Bạn phải gọi backward() trên logit target trước khi explain, "
                    "và trong forward cần gọi self.attn_map.retain_grad()."
                )

            attn_maps.append(attn.detach())       # [B, H, N, N]
            attn_grads.append(attn.grad.detach()) # [B, H, N, N]

        return attn_maps, attn_grads
    
    def _compute_head_weights(
        self,
        attn_grads: List[torch.Tensor],
        cls_index: int = 0,
    ):
        layer_head_weights: List[torch.Tensor] = []

        for grad in attn_grads:
            # grad: [B, H, N, N]
            B, H, N, _ = grad.shape

            # lấy hàng i = cls_index
            grad_cls = grad[:, :, cls_index, :]  # [B, H, N]

            # nếu có class token, bỏ cột đầu tương ứng class
            if self.has_cls:
                grad_cls = grad_cls[..., 1:]     # [B, H, N-1]

            # xử lý gradient theo config
            if self.clamp_grad_pos:
                grad_cls = F.relu(grad_cls)
            if self.use_abs_grad:
                grad_cls = grad_cls.abs()

            # tổng trên patch, rồi trung bình trên batch → [H]
            g_h = grad_cls.sum(dim=-1).mean(dim=0)  # [H]

            g_h = g_h + self.eps
            alpha_h = g_h / g_h.sum()
            layer_head_weights.append(alpha_h)      # [H]

        return layer_head_weights
    
    def _build_rollout_matrix(
        self,
        attn_maps: List[torch.Tensor],
        head_weights: List[torch.Tensor],
        cls_index: int = 0,
    ) -> torch.Tensor:
        # khởi tạo rollout là identity
        B, H, N, _ = attn_maps[0].shape
        device = attn_maps[0].device

        A_rollout = torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]

        for A_l, alpha_h in zip(attn_maps, head_weights):
            # A_l: [B, H, N, N], alpha_h: [H]
            alpha = alpha_h.view(1, -1, 1, 1)        # [1, H, 1, 1]
            A_combo = (A_l * alpha).sum(dim=1)       # [B, N, N]

            # cộng residual
            I = torch.eye(N, device=A_combo.device).unsqueeze(0)  # [1, N, N]
            A_tilde = A_combo + I                                # [B, N, N]

            # rollout
            A_rollout = torch.bmm(A_rollout, A_tilde)            # [B, N, N]

        return A_rollout
    
    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        img_size: Optional[Tuple[int, int]] = None,
        return_rollout: bool = False,
    ):
        """
        Tính giải thích LAGR cho batch ảnh x.

        Args:
            x: Tensor [B, 3, H, W]
            target:
                - None: dùng argmax(logits) cho mỗi mẫu
                - Tensor [B]: index class mục tiêu cho từng mẫu
            img_size:
                - None: suy ra từ grid_size * 16
                - (H_img, W_img): kích thước muốn upsample heatmap
            return_rollout:
                - False (default): chỉ trả về token_scores và heatmap
                - True: trả thêm A_rollout [B, N, N]

        Returns:
            Nếu return_rollout == False:
                token_scores: [B, N-1]
                    - độ quan trọng của từng patch token (bỏ class token)
                heatmap: [B, 1, H_img, W_img]
                    - bản đồ nhiệt đã upsample theo kích thước ảnh

            Nếu return_rollout == True:
                token_scores: [B, N-1]
                heatmap: [B, 1, H_img, W_img]
                A_rollout: [B, N, N]
                    - ma trận rollout full token→token (class row vs mọi token)
        """

        self.model.eval()

        for blk in self.blocks:
            attn_mod = getattr(blk, "attn", None)
            if attn_mod is not None and hasattr(attn_mod, "record_attn"):
                attn_mod.record_attn = True

        self.model.zero_grad(set_to_none=True)
        x = x.to(next(self.model.parameters()).device)

        logits = self.model(x)   # [B, num_classes]
        B, C = logits.shape

        if target is None:
            target = logits.argmax(dim=-1)  # [B]
        else:
            target = target.to(logits.device)

        idx = torch.arange(B, device=logits.device)
        logit_target = logits[idx, target]  # [B]

        logit_target.sum().backward()
        attn_maps, attn_grads = self._collect_attn_and_grads()

        head_weights = self._compute_head_weights(attn_grads, cls_index=0)
        A_rollout = self._build_rollout_matrix(attn_maps, head_weights, cls_index=0)
        cls_row = A_rollout[:, 0, :]

        if self.has_cls:
            token_scores = cls_row[:, 1:]
        else:
            token_scores = cls_row

        token_scores = token_scores - token_scores.amin(dim=1, keepdim=True)
        token_scores = token_scores / (token_scores.amax(dim=1, keepdim=True) + self.eps)

        heatmap = self._tokens_to_heatmap(token_scores, img_size=img_size)

        if return_rollout:
            return token_scores, heatmap, A_rollout
        else:
            return token_scores, heatmap