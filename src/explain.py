import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm

def _get_item(ds, idx):
    s = ds[idx]
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
    x = x.unsqueeze(0).to(device, non_blocking=True)  # [1,3,H,W]
    y = None if y is None else y.view(-1).to(device)
    return x, y

def _denorm_img(x, mean, std):
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    img = x * std_t + mean_t
    return img.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()

def _norm01_quantile(h, q=0.99, eps=1e-6):
    if h.dim() == 3:
        h = h.squeeze(0)
    hp = h.clamp_min(0)
    vmax = torch.quantile(hp.flatten(), q)
    if (not torch.isfinite(vmax)) or vmax <= 0:
        vmax = hp.max()
    vmax = vmax + eps
    return (hp / vmax).clamp(0, 1)

def _overlay(img_hw3, heat_hw, alpha=0.45, cmap="jet"):
    cmap_fn = cm.get_cmap(cmap)
    heat_rgb = cmap_fn(heat_hw.detach().cpu().numpy())[..., :3]  # H×W×3
    overlay = (1 - alpha) * img_hw3 + alpha * heat_rgb
    return overlay

def _tokens_to_heat(rtokens, model, x_shape, has_cls=True):
        """
        rtokens: [1, N] relevance token-level tại input block0
        model: ViT backbone (có patch_embed).
        x_shape: (B,3,H,W) của input x.
        returns: [H, W] heatmap upsample về kích thước ảnh.
        """
        B, C_in, H, W = x_shape
        assert B == 1, "Hàm này hiện tại giả sử batch size = 1."

        r = rtokens[0]  # [N]
        if has_cls:
            r_patch = r[1:]
        else:
            r_patch = r

        Np = r_patch.numel()

        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
            gh, gw = model.patch_embed.grid_size
        else:
            if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
                P = model.patch_embed.proj.kernel_size[0]
                S = model.patch_embed.proj.stride[0]
            else:
                raise RuntimeError("Không tìm được thông tin patch size/grid_size từ model.")
            gh = H // S
            gw = W // S

        assert gh * gw == Np, f"grid_size ({gh},{gw}) không khớp với số patch {Np}"

        r_grid = r_patch.view(gh, gw)  # [gh, gw]

        r_up = F.interpolate(
            r_grid.unsqueeze(0).unsqueeze(0),  # [1,1,gh,gw]
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]  # [H,W]
        return r_up

def visualize_explain(ds, idx, attr_model: nn.Module, use_pred=True, device="cuda", 
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), 
                      alpha=0.45,  q=0.99, has_cls=True, save_path=None):
    attr_model.model.to(device)
    attr_model.model.eval()

    img_t, y = _get_item(ds, idx)
    x, y = _to_device_batch(img_t, y, device)

    if use_pred or (y is None):
        with torch.no_grad():
            logits_pred = attr_model.model(x)
            y_true = logits_pred.argmax(dim=1)
    else:
        y_true = y

    out = attr_model.attribute(x, y_true, return_all_layers=False)
    rtokens = out["rtokens"]

    heat_map = _tokens_to_heat(rtokens, attr_model.model, x.shape, has_cls=has_cls)  # [H,W]
    heat_norm = _norm01_quantile(heat_map, q=q)

    img_np = _denorm_img(img_t, mean, std)    # H×W×3 in [0,1]
    overlay = _overlay(img_np, heat_norm, alpha=alpha, cmap="jet")

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Attribution heatmap (overlay)")

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0, dpi=300)

    plt.show()

    return heat_map.detach().cpu(), overlay