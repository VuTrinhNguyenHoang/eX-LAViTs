import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple
import psutil
import os, math
from tqdm.auto import tqdm

class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = "", fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int] = (1,)) -> list:
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    
    Args:
        output: Model predictions of shape [batch_size, num_classes]
        target: Ground truth labels of shape [batch_size]
        topk: Tuple of k values to compute top-k accuracy for
        
    Returns:
        List of accuracies for each k in topk
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res if len(res) > 1 else res[0]

def top_k_accuracy(output: torch.Tensor, target: torch.Tensor, k: int = 5) -> float:
    """
    Computes top-k accuracy.
    
    Args:
        output: Model predictions of shape [batch_size, num_classes]
        target: Ground truth labels of shape [batch_size]
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as percentage
    """
    acc = accuracy(output, target, topk=(k,))
    return acc[0].item()

def compute_classification_metrics(
    all_preds: np.ndarray,
    all_targets: np.ndarray,
    class_names: Optional[list] = None,
    return_confusion_matrix: bool = False
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        all_preds: Array of predicted class indices
        all_targets: Array of true class indices  
        class_names: List of class names for the report
        return_confusion_matrix: Whether to include confusion matrix
        
    Returns:
        Dictionary containing classification metrics
    """
    
    # Basic accuracy
    total_accuracy = (all_preds == all_targets).mean() * 100
    
    # Classification report
    target_names = class_names if class_names is not None else None
    report = classification_report(
        all_targets, 
        all_preds, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Extract macro and weighted averages
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    
    metrics = {
        'accuracy': total_accuracy,
        'macro_precision': macro_avg['precision'] * 100,
        'macro_recall': macro_avg['recall'] * 100,
        'macro_f1': macro_avg['f1-score'] * 100,
        'weighted_precision': weighted_avg['precision'] * 100,
        'weighted_recall': weighted_avg['recall'] * 100,
        'weighted_f1': weighted_avg['f1-score'] * 100,
        'classification_report': report
    }
    
    # Add confusion matrix if requested
    if return_confusion_matrix:
        cm = confusion_matrix(all_targets, all_preds)
        metrics['confusion_matrix'] = cm
    
    return metrics

def compute_loss_and_accuracy(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Compute loss and accuracy for a dataset.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for the dataset
        criterion: Loss criterion
        device: Device to run computation on
        return_predictions: Whether to return predictions and targets
        
    Returns:
        Dictionary containing loss, accuracy, and optionally predictions
    """
    model.eval()
    
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            images, target = images.to(device), target.to(device)
            
            # Compute output
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1, ))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            
            if return_predictions:
                _, preds = torch.max(output, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
    results = {
        'loss': losses.avg,
        'top1_accuracy': top1.avg
    }
    
    if return_predictions:
        results['predictions'] = np.array(all_preds)
        results['targets'] = np.array(all_targets)
    
    return results

class MetricsTracker:
    """Track metrics over training epochs."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []
        self.memory_usage = []
    
    def update(self, 
               train_loss: float = None,
               val_loss: float = None,
               train_accuracy: float = None,
               val_accuracy: float = None,
               lr: float = None,
               epoch_time: float = None,
               memory_usage: float = None):
        """Update metrics for current epoch."""
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_accuracy is not None:
            self.train_accuracies.append(train_accuracy)
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy)
        if lr is not None:
            self.learning_rates.append(lr)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
        if memory_usage is not None:
            self.memory_usage.append(memory_usage)
    
    def get_best_epoch(self, metric: str = 'val_loss') -> int:
        """Get epoch with best performance for given metric."""
        if metric == 'val_loss' and self.val_losses:
            return np.argmin(self.val_losses)
        elif metric == 'train_loss' and self.train_losses:
            return np.argmin(self.train_losses)
        elif metric == 'val_acc' and self.val_accuracies:
            return np.argmax(self.val_accuracies)
        elif metric == 'train_acc' and self.train_accuracies:
            return np.argmax(self.train_accuracies)
        else:
            return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        summary = {}
        
        if self.train_losses:
            summary['best_train_loss'] = min(self.train_losses)
            summary['final_train_loss'] = self.train_losses[-1]
        
        if self.val_losses:
            summary['best_val_loss'] = min(self.val_losses)
            summary['final_val_loss'] = self.val_losses[-1]
            summary['best_val_loss_epoch'] = self.get_best_epoch('val_loss')
        
        if self.train_accuracies:
            summary['best_train_accuracy'] = max(self.train_accuracies)
            summary['final_train_accuracy'] = self.train_accuracies[-1]
        
        if self.val_accuracies:
            summary['best_val_accuracy'] = max(self.val_accuracies)
            summary['final_val_accuracy'] = self.val_accuracies[-1]
            summary['best_val_accuracy_epoch'] = self.get_best_epoch('val_acc')
        
        if self.epoch_times:
            summary['total_training_time'] = sum(self.epoch_times)
            summary['avg_epoch_time'] = np.mean(self.epoch_times)
            summary['min_epoch_time'] = min(self.epoch_times)
            summary['max_epoch_time'] = max(self.epoch_times)
        
        if self.memory_usage:
            summary['peak_memory_usage'] = max(self.memory_usage)
            summary['avg_memory_usage'] = np.mean(self.memory_usage)
        
        return summary

def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # Convert to MB
    except ImportError:
        # Fallback for systems without psutil
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # GPU memory in MB
        else:
            return 0.0  # Unable to get memory info

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple
import psutil
import os, math
from tqdm.auto import tqdm

def normalize_heatmap(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if h.dim() == 3:
        # nếu có kênh, tổng theo kênh
        if h.size(0) > 1:
            h = h.sum(dim=0)
        else:
            h = h.squeeze(0)
    h = h.clamp_min(0)
    s = h.sum()
    if s <= eps:
        # nếu toàn 0, trả phân bố đều
        H, W = h.shape
        return torch.full_like(h, 1.0 / (H * W))
    return h / s

def heatmap_entropy(h: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Entropy chuẩn hoá về [0,1].
    h: [H,W] hoặc [C,H,W] (sẽ normalize trước).
    """
    p = normalize_heatmap(h, eps)
    N = p.numel()
    ent = -(p * (p + eps).log()).sum()
    ent = ent.item()
    ent_norm = ent / math.log(N + eps)
    return float(ent_norm)

def heatmap_gini(h: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Hệ số Gini của phân bố heatmap (0: đều, 1: cực kỳ tập trung).
    h: [H,W] hoặc [C,H,W].
    """
    p = normalize_heatmap(h, eps).view(-1)
    if p.numel() == 0:
        return 0.0
    # sort tăng dần
    p_sorted, _ = torch.sort(p)
    N = p_sorted.numel()
    index = torch.arange(1, N + 1, device=p_sorted.device, dtype=p_sorted.dtype)
    num = (2 * index * p_sorted).sum()
    den = N * p_sorted.sum() + eps
    gini = (num / den - (N + 1) / N).item()
    return float(gini)

@torch.no_grad()
def _perturb_image(x: torch.Tensor,
                   baseline: torch.Tensor,
                   mask: torch.Tensor,
                   mode: str) -> torch.Tensor:
    """
    x, baseline: [1,3,H,W]
    mask: [1,1,H,W] với giá trị 0/1
    mode: 'deletion' hoặc 'insertion'
    """
    if mode == "deletion":
        return mask * x + (1 - mask) * baseline
    elif mode == "insertion":
        return (1 - mask) * baseline + mask * x
    else:
        raise ValueError("mode phải là 'deletion' hoặc 'insertion'")
    
@torch.no_grad()
def deletion_insertion_auc(
    model: torch.nn.Module,
    x: torch.Tensor,              # [1,3,H,W]
    y_true: torch.Tensor,         # [1]
    heatmap: torch.Tensor,        # [H,W] hoặc [C,H,W]
    steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Tính AUC–Deletion và AUC–Insertion cho MỘT ảnh.

    - Deletion: bắt đầu từ ảnh gốc, lần lượt thay top-pixel theo heatmap bằng baseline.
      -> AUC càng nhỏ càng tốt.
    - Insertion: bắt đầu từ baseline, lần lượt chèn top-pixel từ ảnh gốc.
      -> AUC càng lớn càng tốt.

    Trả về: {'auc_del': float, 'auc_ins': float}
    """
    assert x.size(0) == 1, "Hàm này hiện thiết kế cho batch=1."
    if device is not None:
        x = x.to(device)
        y_true = y_true.to(device)
        if baseline is not None:
            baseline = baseline.to(device)
    else:
        device = x.device

    B, C, H, W = x.shape
    if baseline is None:
        baseline = torch.zeros_like(x)  # ảnh nền = 0 (tương ứng mean nếu đã normalize)

    # dùng positive heatmap, flatten
    if heatmap.dim() == 3 and heatmap.size(0) > 1:
        h = heatmap.sum(dim=0)
    else:
        h = heatmap.squeeze(0) if heatmap.dim() == 3 else heatmap
    h = h.clamp_min(0)
    flat = h.view(-1)
    N = flat.numel()
    if steps > N:
        steps = N

    # thứ tự pixel từ quan trọng -> kém
    order = torch.argsort(flat, descending=True)

    # số pixel thao tác mỗi bước (step cuối có thể ít hơn)
    k = N // steps

    def _scores_along_path(mode: str) -> np.ndarray:
        scores = []
        # mask khởi điểm
        if mode == "deletion":
            mask_flat = torch.ones(N, device=device)
        else:  # insertion
            mask_flat = torch.zeros(N, device=device)

        for s in range(steps + 1):
            mask = mask_flat.view(1, 1, H, W)
            x_pert = _perturb_image(x, baseline, mask, mode)
            logits = model(x_pert)
            score = logits.gather(1, y_true[:, None]).squeeze(1)  # [1]
            scores.append(score.item())

            if s == steps:
                break

            # cập nhật mask cho bước tiếp theo
            start = s * k
            end = (s + 1) * k if s < steps - 1 else N
            idx = order[start:end]
            if mode == "deletion":
                mask_flat[idx] = 0.0
            else:
                mask_flat[idx] = 1.0

        return np.array(scores, dtype=np.float64)

    # Deletion: chuẩn hoá về [0,1] với điểm đầu = 1
    scores_del = _scores_along_path("deletion")
    if abs(scores_del[0]) < 1e-8:
        scores_del_norm = scores_del  # tránh chia 0, curve gần như flat
    else:
        scores_del_norm = scores_del / (scores_del[0] + 1e-8)

    # Insertion: chuẩn hoá sao cho baseline -> 0, full -> 1
    scores_ins = _scores_along_path("insertion")
    s0, sT = scores_ins[0], scores_ins[-1]
    denom = (sT - s0)
    if abs(denom) < 1e-8:
        scores_ins_norm = scores_ins - s0  # gần như 0 hết
    else:
        scores_ins_norm = (scores_ins - s0) / (denom + 1e-8)

    xs = np.linspace(0.0, 1.0, len(scores_del_norm))
    auc_del = float(np.trapz(scores_del_norm, xs))
    auc_ins = float(np.trapz(scores_ins_norm, xs))

    return {"auc_del": auc_del, "auc_ins": auc_ins}

def _extract_heatmap_from_output(out: Dict[str, torch.Tensor],
                                 prefer: str = "rtokens_up") -> torch.Tensor:
    """
    out: dict trả về từ attributor.attribute
    prefer:
      - 'rtokens_up': dùng H×W
      - 'rpix_sum' : nếu có 'rpix', dùng tổng kênh
    """
    if prefer == "rtokens_up" and ("rtokens_up" in out):
        return out["rtokens_up"]       # [B,H,W]
    if prefer == "rpix_sum" and ("rpix" in out):
        return out["rpix"].sum(dim=1)  # [B,H,W]
    # fallback: ưu tiên rtokens_up rồi rpix
    if "rtokens_up" in out:
        return out["rtokens_up"]
    if "rpix" in out:
        return out["rpix"].sum(dim=1)
    raise KeyError("Output không có 'rtokens_up' hoặc 'rpix'.")

@torch.no_grad()
def evaluate_attributor(
    attributor,
    backbone: torch.nn.Module,
    dataloader,
    device: str = "cuda",
    steps: int = 50,
    use_pred: bool = True,
    prefer_map: str = "rtokens_up",
    max_batches: int | None = None
) -> Dict[str, float]:
    """
    Đánh giá một phương pháp attribution trên toàn bộ dataloader.

    - attributor: SSRP, ViTGradCAM, Occlusion, KernelSHAP,
                  IntegratedGradients, Rollout, ...
      (phải có thuộc tính `.model` trỏ tới backbone).

    - backbone: model phân loại (thường chính là attributor.model).

    - dataloader: yield (x, y) hoặc dict{'image','label',...}.

    Trả về: dict với mean của:
      'auc_del', 'auc_ins', 'entropy', 'gini', 'n_samples'.
    """

    backbone.to(device).eval()
    attributor.model.to(device).eval()

    auc_del_list, auc_ins_list, ent_list, gini_list = [], [], [], []
    n_samples = 0
    n_batches = 0

    for batch in tqdm(dataloader, total=len(dataloader)):
        n_batches += 1
        if (max_batches is not None) and (n_batches > max_batches):
            break

        # 1) lấy x, y
        if isinstance(batch, (list, tuple)):
            x, y = batch[0], batch[1]
        elif isinstance(batch, dict):
            x = None; y = None
            for k in ("image", "img", "tensor", "x"):
                if k in batch: 
                    x = batch[k]; break
            for k in ("label", "y", "target"):
                if k in batch: 
                    y = batch[k]; break
            if x is None:
                raise ValueError("Không tìm thấy key ảnh trong batch dict.")
        else:
            raise TypeError("Batch phải là (x,y) hoặc dict.")

        x = x.to(device)
        if y is not None:
            y = y.to(device)

        B = x.size(0)

        # 2) tính y_true cho cả batch
        if use_pred or y is None:
            logits = backbone(x)
            y_true_all = logits.argmax(dim=1)
        else:
            y_true_all = y.view(-1)

        # 3) duyệt từng ảnh trong batch (attribute luôn với B=1)
        for i in range(B):
            xi = x[i:i+1]              # [1,3,H,W]
            yi = y_true_all[i:i+1]     # [1]

            out = attributor.attribute(xi, yi)
            hi_all = _extract_heatmap_from_output(out, prefer=prefer_map)  # [1,H,W]
            hi = hi_all[0]             # [H,W]

            di = deletion_insertion_auc(backbone, xi, yi, hi, steps=steps)
            ent = heatmap_entropy(hi)
            gi  = heatmap_gini(hi)

            auc_del_list.append(di["auc_del"])
            auc_ins_list.append(di["auc_ins"])
            ent_list.append(ent)
            gini_list.append(gi)
            n_samples += 1

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "auc_del_mean": _mean(auc_del_list),
        "auc_ins_mean": _mean(auc_ins_list),
        "entropy_mean": _mean(ent_list),
        "gini_mean": _mean(gini_list),
        "n_samples": n_samples,
    }