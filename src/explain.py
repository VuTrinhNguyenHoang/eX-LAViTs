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
    attr_dict: OrderedDict,
    use_pred: bool,
    device: str,
    mean=(0.485,0.456,0.406),
    std=(0.229,0.224,0.225),
    alpha: float = 0.45,
    cmap_name: str = "jet",
    q: float = 0.99
):
    """
    ds: dataset, phần tử ds[idx] -> (img, label) hoặc dict.
    attr_dict: OrderedDict tên_phương_pháp -> Attributor (SSRP, IG, ...)

    Mỗi attributor nên có:
        attribute(x, y_true) -> dict với:
          - 'rtokens_up': [B,H,W] (ưu tiên cho trực quan)
          - (hoặc) 'rtokens': [B,Hn,Wn]  -> sẽ tự upsample
          - (fallback) 'rpix': [B,3,H,W]
    """

    # 1) lấy mẫu
    img_t, y = _get_item(ds, idx)
    x, y = _to_device_batch(img_t, y, device)

    # 2) y_true từ backbone đầu tiên
    first_attr = next(iter(attr_dict.values()))
    backbone = getattr(first_attr, "model", None)
    if backbone is None:
        raise RuntimeError("Các Attributor cần có thuộc tính .model để suy ra y_true.")
    backbone.to(device).eval()
    with torch.no_grad():
        logits_pred = backbone(x)
        pred_cls = logits_pred.argmax(dim=1)
    y_true = pred_cls if (use_pred or y is None) else y

    # 3) ảnh gốc denorm
    img_np = _denorm_img(img_t.to(device), mean, std)

    # 4) thu heatmap (pixel-level, dùng cho overlay) từ từng phương pháp
    heatmaps = OrderedDict()
    B, _, H, W = x.shape

    for name, attr in attr_dict.items():
        attr.model.to(device).eval()
        out = attr.attribute(x, y_true)

        if "rtokens_up" in out:
            # đã có upsample sẵn H×W
            h = out["rtokens_up"][0]      # [H,W]
        elif "rtokens" in out:
            # chỉ có token map -> tự upsample
            rtok = out["rtokens"][0]      # [Hn,Wn]
            cam = rtok.unsqueeze(0).unsqueeze(0)  # [1,1,Hn,Wn]
            h_up = F.interpolate(cam, size=(H, W), mode="bilinear",
                                  align_corners=False)[0,0]
            h = h_up
        elif "rpix" in out:
            # fallback: pixel-level theo kênh
            h = out["rpix"][0].sum(dim=0)  # [H,W]
        else:
            raise RuntimeError(f"{name}: không tìm thấy 'rtokens_up', 'rtokens' hoặc 'rpix' trong output.")

        heatmaps[name] = _norm01_quantile(h, q=q)

    # 5) vẽ lưới
    n_methods = len(heatmaps)
    n_plots = n_methods + 1
    ncols = min(4, n_plots)
    nrows = int(math.ceil(n_plots / ncols))

    fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axs = np.array(axs).reshape(nrows, ncols)

    # ảnh gốc
    axs[0,0].imshow(img_np)
    axs[0,0].set_title("Original")
    axs[0,0].axis("off")

    # các heatmap
    i = 1
    for name, h in heatmaps.items():
        r = i // ncols
        c = i % ncols
        axs[r, c].imshow(_overlay(img_np, h, alpha=alpha, cmap=cmap_name))
        axs[r, c].set_title(name)
        axs[r, c].axis("off")
        i += 1

    # ẩn ô dư
    while i < nrows * ncols:
        r = i // ncols
        c = i % ncols
        axs[r, c].axis("off")
        i += 1

    plt.tight_layout()
    plt.show()

    return heatmaps
