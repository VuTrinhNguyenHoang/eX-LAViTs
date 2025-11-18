import torch
import torch.nn.functional as F
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import math

def _get_item(ds, idx):
    s = ds[idx]
    img, y = None, None
    if isinstance(s, (tuple, list)):
        img = s[0]
        if len(s) > 1:
            y = s[1]
    elif isinstance(s, dict):
        for k in ("image", "img", "tensor", "x"):
            if k in s:
                img = s[k]
                break
        for k in ("label", "y", "target"):
            if k in s:
                y = s[k]
                break
    else:
        img = s
    if not torch.is_tensor(img):
        raise TypeError("Dataset phải trả về ảnh dạng torch.Tensor sau transform.")
    if y is not None and not torch.is_tensor(y):
        y = torch.tensor(y)
    return img, y

def _to_device_batch(x, y, device):
    x = x.unsqueeze(0).to(device, non_blocking=True)
    y = None if y is None else y.view(-1).to(device)
    return x, y

def _denorm_img(x, mean, std):
    mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1,1,1)
    std  = torch.tensor(std,  dtype=x.dtype, device=x.device).view(-1,1,1)
    img = x * std + mean
    return img.clamp(0,1).permute(1,2,0).detach().cpu().numpy()

def _norm01_quantile(h, q=0.99, eps=1e-6):
    if h.dim() == 3:
        h = h.squeeze(0)
    hp = h.clamp_min(0)
    flat = hp.flatten()
    if flat.numel() == 0:
        return torch.zeros_like(h)
    vmax = torch.quantile(flat, q)
    if (not torch.isfinite(vmax)) or vmax <= 0:
        vmax = flat.max()
    vmax = vmax + eps
    return (hp / vmax).clamp(0,1)

def _overlay(img_hw3, heat_hw, alpha=0.45, cmap="jet"):
    cmap_fn = cm.get_cmap(cmap)
    heat_rgb = cmap_fn(heat_hw.detach().cpu().numpy())[...,:3]
    overlay = (1 - alpha) * img_hw3 + alpha * heat_rgb
    return overlay

def visualize_methods(
    ds,
    idx,
    attr_dict,
    use_pred: bool,
    device: str,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    alpha: float = 0.45,
    cmap_name: str = "jet",
    q: float = 0.99,
):
    # Đảm bảo thứ tự cố định
    if not isinstance(attr_dict, OrderedDict):
        attr_dict = OrderedDict(attr_dict)

    # 1) Lấy mẫu
    img_t, y = _get_item(ds, idx)
    x, y = _to_device_batch(img_t, y, device)

    # 2) Backbone từ attributor đầu tiên
    first_attr = next(iter(attr_dict.values()))
    backbone = getattr(first_attr, "model", None)
    if backbone is None:
        raise RuntimeError("Các Attributor cần có thuộc tính .model để suy ra y_true.")

    backbone.to(device).eval()
    with torch.no_grad():
        logits_pred = backbone(x)
        pred_cls = logits_pred.argmax(dim=1)

    y_true = pred_cls if (use_pred or y is None) else y

    # 3) Ảnh gốc (denorm) để overlay
    img_np = _denorm_img(img_t.to(device), mean, std)

    # 4) Thu heatmap từ từng phương pháp
    heatmaps = OrderedDict()
    B, _, H, W = x.shape

    for name, attr in attr_dict.items():
        # Đảm bảo attributor và backbone trên đúng device
        if hasattr(attr, "model"):
            attr.model.to(device).eval()

        # Gọi attribute: tất cả method mới đều hỗ trợ (x, target=..., img_size=...)
        token_scores, heatmap = attr.attribute(
            x,
            target=y_true,
            img_size=(H, W),
        )
        # heatmap: [B,1,H,W]
        h = heatmap[0, 0]  # [H,W]
        h_norm = _norm01_quantile(h, q=q)
        heatmaps[name] = h_norm

    # 5) Vẽ lưới
    n_methods = len(heatmaps)
    n_plots = n_methods + 1  # +1 cho ảnh gốc
    ncols = min(4, n_plots)
    nrows = int(math.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axs = np.array(axs).reshape(nrows, ncols)

    # Ảnh gốc
    axs[0, 0].imshow(img_np)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis("off")

    # Các heatmap
    i = 1
    for name, h in heatmaps.items():
        r = i // ncols
        c = i % ncols
        axs[r, c].imshow(_overlay(img_np, h, alpha=alpha, cmap=cmap_name))
        axs[r, c].set_title(name)
        axs[r, c].axis("off")
        i += 1

    # Ẩn ô dư
    while i < nrows * ncols:
        r = i // ncols
        c = i % ncols
        axs[r, c].axis("off")
        i += 1

    plt.tight_layout()
    plt.show()

    return heatmaps

