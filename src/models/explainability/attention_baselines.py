import torch

from .linear_rollout import tokens_to_heatmap


def set_record_attention(model, record):
    for module in model.modules():
        if hasattr(module, "record_attn") and hasattr(module, "attn_map"):
            module.record_attn = record
            module.attn_map = None


class CDAM:
    def __init__(
        self, model, has_cls=True, blocks_attr="blocks", layer="last", eps=1e-6, text=False
    ):
        self.model = model.eval()
        self.blocks = list(getattr(model, blocks_attr))
        self.has_cls = has_cls
        self.layer = layer
        self.eps = eps
        self.text = text

    def _combine(self, maps):
        selected_maps = [maps[-1]] if self.layer == "last" else maps
        layer_rels = []
        for attn, grad in selected_maps:
            matrix = (attn * grad).clamp_min(0.0).mean(dim=1)
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + self.eps)
            relevance = matrix[:, 0, 1:] if self.has_cls else matrix.mean(dim=1)
            layer_rels.append(relevance)
        return torch.stack(layer_rels, dim=0).mean(dim=0)

    def attribute(self, x, target=None, use_logits=True):
        self.model.zero_grad(set_to_none=True)
        set_record_attention(self.model, True)
        try:
            logits = self.model(x)
            if target is None:
                target = logits.argmax(dim=-1)
            values = logits if use_logits else logits.softmax(dim=-1)
            values.gather(1, target.reshape(-1, 1)).sum().backward()
            maps = []
            for block in self.blocks:
                attn = block.attn.attn_map
                if attn is not None and attn.grad is not None:
                    maps.append((attn, attn.grad))
            if not maps:
                raise RuntimeError("No attention maps/gradients collected.")
            relevance = self._combine(maps).clamp_min(0.0)
            relevance = relevance / (relevance.amax(dim=-1, keepdim=True) + self.eps)
            heatmap = (
                relevance.unsqueeze(1)
                if self.text
                else tokens_to_heatmap(relevance, self.model)
            )
            return relevance.detach(), heatmap.detach()
        finally:
            set_record_attention(self.model, False)
            self.model.zero_grad(set_to_none=True)


class GMAR(CDAM):
    def __init__(
        self,
        model,
        has_cls=True,
        blocks_attr="blocks",
        eps=1e-6,
        use_residual=True,
        text=False,
    ):
        super().__init__(model, has_cls, blocks_attr, eps=eps, text=text)
        self.use_residual = use_residual

    def _head_weighted_attention(self, attn, grad):
        head_score = (attn * grad).mean(dim=(-1, -2)).clamp_min(0.0)
        denominator = head_score.sum(dim=1, keepdim=True)
        uniform = torch.full_like(head_score, 1.0 / head_score.shape[1])
        weights = torch.where(
            denominator > self.eps, head_score / (denominator + self.eps), uniform
        )
        matrix = (weights[:, :, None, None] * attn).sum(dim=1).clamp_min(0.0)
        return matrix / (matrix.sum(dim=-1, keepdim=True) + self.eps)

    def _combine(self, maps):
        result = None
        for attn, grad in maps:
            matrix = self._head_weighted_attention(attn, grad)
            if self.use_residual:
                matrix = matrix + torch.eye(matrix.shape[-1], device=matrix.device).unsqueeze(
                    0
                )
                matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + self.eps)
            result = matrix if result is None else torch.bmm(result, matrix)
        return result[:, 0, 1:] if self.has_cls else result.mean(dim=1)


class GradientRollout(CDAM):
    def _combine(self, maps):
        result = None
        for attn, grad in maps:
            matrix = (attn * grad).clamp_min(0.0).mean(dim=1)
            matrix = matrix + torch.eye(matrix.shape[-1], device=matrix.device).unsqueeze(0)
            matrix = matrix / matrix.sum(dim=-1, keepdim=True)
            result = matrix if result is None else torch.bmm(matrix, result)
        return result[:, 0, 1:] if self.has_cls else result.mean(dim=1)
