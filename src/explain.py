import torch
import matplotlib.pyplot as plt
import numpy as np

def denorm_image(
    img: torch.Tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    img: [C,H,W], tensor trên CPU, đã normalize theo mean/std.
    Trả về ảnh numpy [H,W,3] trong [0,1].
    """
    if isinstance(mean, (tuple, list)):
        mean = torch.tensor(mean).view(-1, 1, 1)
    if isinstance(std, (tuple, list)):
        std = torch.tensor(std).view(-1, 1, 1)

    x = img.clone()
    x = x * std + mean
    x = x.clamp(0.0, 1.0)
    x = x.permute(1, 2, 0)  # [H,W,C]
    return x.numpy()

def visualize_methods(
    dataset,
    idx: int,
    methods: dict,
    use_pred: bool = True,
    device: str = "cuda",
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    class_names=None,
    vmax: float = 1.0,
    save_path: str = None,
):
    """
    dataset: torch.utils.data.Dataset trả (image, label, ...) hoặc (image, label).
    idx: index trong dataset.
    methods: dict tên_phương_pháp -> attributor (có .attribute() và .model).
    use_pred: True → giải thích theo lớp dự đoán; False → dùng label GT.
    device: 'cuda' hoặc 'cpu'.
    mean/std: dùng để de-normalize ảnh.
    class_names: list tên class (tùy chọn).
    vmax: max value cho heatmap (0–vmax). Mặc định 1.0.
    save_path: nếu != None thì lưu figure ra file.
    """
    # Lấy mẫu từ dataset
    sample = dataset[idx]
    if isinstance(sample, (tuple, list)):
        img = sample[0]
        label = sample[1]
    else:
        img = sample
        label = None

    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)

    # Đảm bảo shape [C,H,W]
    if img.ndim == 4:
        img = img[0]
    assert img.ndim == 3, f"Expect image [C,H,W], got {img.shape}"

    if isinstance(label, torch.Tensor):
        label = int(label.item())
    elif label is not None:
        label = int(label)

    # Chuẩn bị input cho model
    x = img.unsqueeze(0).to(device)  # [1,3,H,W]

    # Chọn model base để lấy logits/pred
    first_attr = next(iter(methods.values()))
    base_model = getattr(first_attr, "model", None)

    pred_label = None
    pred_prob = None

    if base_model is not None:
        base_model.eval()
        with torch.no_grad():
            logits = base_model(x)  # [1,C]
            probs = logits.softmax(dim=-1)
            pred_label = int(probs.argmax(dim=-1).item())
            pred_prob = float(probs.max().item())

    # Quyết định target cho attributors
    if use_pred or label is None:
        target_tensor = None  # để attributor tự argmax bên trong
        label_to_show = pred_label
    else:
        target_tensor = torch.tensor([label], device=device, dtype=torch.long)
        label_to_show = label

    # Gọi từng phương pháp để lấy heatmap
    heatmaps = {}  # name -> [H,W] numpy
    for name, attr in methods.items():
        # Không dùng torch.no_grad() ở đây vì nhiều attributor cần backward
        patch_rel, hm = attr.attribute(x, target=target_tensor, use_logits=True)
        # hm: [1,1,H,W]
        hm_2d = hm[0, 0].detach().cpu().numpy()
        heatmaps[name] = hm_2d

    # Chuẩn bị ảnh gốc (ở CPU)
    img_vis = denorm_image(img.cpu(), mean=mean, std=std)

    # Bố cục figure
    n_methods = len(methods)
    n_plots = n_methods + 1  # +1 cho ảnh gốc
    n_cols = min(4, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.reshape(-1)

    # Vẽ ảnh gốc ở ô đầu
    ax0 = axes[0]
    ax0.imshow(img_vis)
    title0 = "Original"
    if label_to_show is not None:
        if class_names is not None and 0 <= label_to_show < len(class_names):
            title0 += f"\nlabel={class_names[label_to_show]}"
        else:
            title0 += f"\nlabel={label_to_show}"
    if pred_prob is not None:
        title0 += f"  p={pred_prob:.2f}"
    ax0.set_title(title0)
    ax0.axis("off")

    # Vẽ từng phương pháp
    for ax, (name, hm) in zip(axes[1:], heatmaps.items()):
        ax.imshow(img_vis)
        ax.imshow(hm, cmap="jet", alpha=0.5, vmin=0.0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")

    # Ẩn các ô dư
    for ax in axes[1 + len(heatmaps):]:
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig