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

def to_token_heatmap(
    heat_hw: torch.Tensor,        # [H,W] hoặc [1,H,W]
    backbone: torch.nn.Module,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Chuyển heatmap pixel H×W về heatmap theo patch/token Hn×Wn
    bằng average pooling theo stride patch_embed.

    Trả về: [Hn, Wn]
    """
    if heat_hw.dim() == 3:
        heat_hw = heat_hw.squeeze(0)
    H, W = heat_hw.shape

    pe = backbone.patch_embed.proj
    S = pe.stride[0]
    assert H % S == 0 and W % S == 0, "H,W phải chia hết cho stride patch."

    h = heat_hw.unsqueeze(0).unsqueeze(0)         # [1,1,H,W]
    h_tok = F.avg_pool2d(h, kernel_size=S, stride=S)  # [1,1,Hn,Wn]
    h_tok = h_tok[0, 0]                           # [Hn,Wn]

    h_tok = h_tok.clamp_min(0)
    if h_tok.sum() <= eps:
        Hn, Wn = h_tok.shape
        return torch.full_like(h_tok, 1.0 / (Hn * Wn))
    return h_tok

@torch.no_grad()
def deletion_insertion_auc_tokens(
    model: torch.nn.Module,
    x: torch.Tensor,              # [1,3,H,W]
    y_true: torch.Tensor,         # [1]
    heat_tok: torch.Tensor,       # [Hn,Wn] token-level
    steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    AUC Deletion/Insertion patch-level:
      - heat_tok: relevance trên grid Hn×Wn (mỗi cell = 1 patch).
      - mask thao tác theo patch, rồi upsample mask lên H×W.

    Score = softmax prob(y_true) -> AUC ∈ [0,1].
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
        baseline = torch.zeros_like(x)

    Hn, Wn = heat_tok.shape
    flat = heat_tok.view(-1).clamp_min(0)    # [Np]
    Np = flat.numel()
    if steps > Np:
        steps = Np
    k = max(1, Np // steps)

    order = torch.argsort(flat, descending=True)

    def _scores_path(mode: str) -> np.ndarray:
        scores = []
        if mode == "deletion":
            mask_tok_flat = torch.ones(Np, device=device)
        else:
            mask_tok_flat = torch.zeros(Np, device=device)

        for s in range(steps + 1):
            mask_tok = mask_tok_flat.view(1, 1, Hn, Wn)        # [1,1,Hn,Wn]
            mask_px = F.interpolate(mask_tok, size=(H, W), mode="nearest")
            x_pert = mask_px * x + (1 - mask_px) * baseline    # [1,3,H,W]

            logits = model(x_pert)
            probs = torch.softmax(logits, dim=1)
            score = probs.gather(1, y_true[:, None]).squeeze(1)
            scores.append(score.item())

            if s == steps:
                break

            start = s * k
            end = (s + 1) * k if s < steps - 1 else Np
            idx = order[start:end]
            if mode == "deletion":
                mask_tok_flat[idx] = 0.0
            else:
                mask_tok_flat[idx] = 1.0

        return np.array(scores, dtype=np.float64)

    scores_del = _scores_path("deletion")
    scores_ins = _scores_path("insertion")

    p_full = scores_del[0]
    p_base = scores_ins[0]

    if p_full < 1e-8:
        scores_del_norm = np.zeros_like(scores_del)
    else:
        scores_del_norm = np.clip(scores_del / (p_full + 1e-8), 0.0, 1.0)

    denom = p_full - p_base
    if abs(denom) < 1e-8:
        scores_ins_norm = np.zeros_like(scores_ins)
    else:
        scores_ins_norm = np.clip((scores_ins - p_base) / (denom + 1e-8), 0.0, 1.0)

    xs = np.linspace(0.0, 1.0, len(scores_del_norm))
    auc_del = float(np.trapz(scores_del_norm, xs))
    auc_ins = float(np.trapz(scores_ins_norm, xs))

    return {"auc_del": auc_del, "auc_ins": auc_ins}

def normalize_heatmap(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # h: [Hn,Wn]
    if h.dim() == 3:
        if h.size(0) > 1:
            h = h.sum(dim=0)
        else:
            h = h.squeeze(0)
    h = h.clamp_min(0)
    s = h.sum()
    if s <= eps:
        H, W = h.shape
        return torch.full_like(h, 1.0 / (H * W))
    return h / s

def heatmap_entropy(h: torch.Tensor, eps: float = 1e-8) -> float:
    p = normalize_heatmap(h, eps)
    N = p.numel()
    ent = -(p * (p + eps).log()).sum().item()
    ent_norm = ent / math.log(N + eps)
    return float(ent_norm)

def heatmap_gini(h: torch.Tensor, eps: float = 1e-8) -> float:
    p = normalize_heatmap(h, eps).view(-1)
    if p.numel() == 0:
        return 0.0
    p_sorted, _ = torch.sort(p)
    N = p_sorted.numel()
    idx = torch.arange(1, N+1, device=p_sorted.device, dtype=p_sorted.dtype)
    num = (2 * idx * p_sorted).sum()
    den = N * p_sorted.sum() + eps
    gini = (num / den - (N + 1) / N).item()
    return float(gini)

def _extract_heatmap_from_output(out: Dict[str, torch.Tensor],
                                 prefer: str = "rtokens_up") -> torch.Tensor:
    if prefer == "rtokens_up" and ("rtokens_up" in out):
        return out["rtokens_up"]       # [B,H,W]
    if prefer == "rpix_sum" and ("rpix" in out):
        return out["rpix"].sum(dim=1)  # [B,H,W]
    if "rtokens_up" in out:
        return out["rtokens_up"]
    if "rpix" in out:
        return out["rpix"].sum(dim=1)
    raise KeyError("Output không có 'rtokens_up' hoặc 'rpix'.")

@torch.no_grad()
def evaluate_attributor_token(
    attributor,
    backbone: torch.nn.Module,
    dataloader,
    device: str = "cuda",
    steps: int = 50,
    use_pred: bool = True,
    prefer_map: str = "rtokens_up",   # dùng khi phải fallback từ pixel
    max_batches: int | None = None
) -> Dict[str, float]:

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

        # 2) y_true cho batch
        if use_pred or y is None:
            logits = backbone(x)
            y_true_all = logits.argmax(dim=1)
        else:
            y_true_all = y.view(-1)

        # 3) duyệt từng ảnh
        for i in range(B):
            xi = x[i:i+1]              # [1,3,H,W]
            yi = y_true_all[i:i+1]     # [1]

            out = attributor.attribute(xi, yi)

            # ưu tiên: token map trực tiếp
            if "rtokens" in out:
                hi_tok = out["rtokens"][0]          # [Hn,Wn]
            else:
                # fallback: lấy pixel map rồi downsample về token
                hi_all = _extract_heatmap_from_output(out, prefer=prefer_map)  # [1,H,W]
                hi_pix = hi_all[0]                    # [H,W]
                hi_tok = to_token_heatmap(hi_pix, backbone)  # [Hn,Wn]

            di = deletion_insertion_auc_tokens(
                model=backbone,
                x=xi,
                y_true=yi,
                heat_tok=hi_tok,
                steps=steps,
                baseline=None,
                device=device,
            )
            ent = heatmap_entropy(hi_tok)
            gi  = heatmap_gini(hi_tok)

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

