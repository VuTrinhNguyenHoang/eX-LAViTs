import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Optional, Tuple
import psutil
from collections import defaultdict
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

def get_patch_grid(model) -> Tuple[int, int]:
    pe = getattr(model, "patch_embed", None)
    if pe is None or getattr(pe, "grid_size", None) is None:
        raise RuntimeError("Model không có patch_embed.grid_size.")
    Hp, Wp = pe.grid_size
    return int(Hp), int(Wp)

def compute_entropy_and_gini(
    patch_rel: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    patch_rel: [N_patches], không nhất thiết chuẩn hoá.
    Trả về:
        entropy (natural log, normalized bởi log N),
        gini (1 - sum p^2).
    """
    # đảm bảo không âm
    scores = patch_rel.clone().detach().float()
    scores = scores.clamp_min(0.0)

    ssum = scores.sum()
    if ssum <= 0:
        # nếu tất cả 0, coi như phân bố đều
        N = scores.numel()
        p = torch.full_like(scores, 1.0 / N)
    else:
        p = scores / ssum  # phân bố xác suất

    # entropy
    logp = torch.log(p + eps)
    ent = -(p * logp).sum().item()
    N = p.numel()
    # chuẩn hoá về [0,1]
    ent_norm = ent / (np.log(N) + eps)

    # gini (1 - sum p^2)
    gini = 1.0 - (p * p).sum().item()
    return float(ent_norm), float(gini)

def auc_trapezoid(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    xs: [T], monotonically increasing (0→1)
    ys: [T], giá trị score đã normalize (0→1)
    """
    return float(np.trapz(ys, xs))

def apply_patch_mask_to_image(
    x: torch.Tensor,           # [B,3,H,W]
    patch_mask: torch.Tensor,  # [B,1,Hp,Wp], 1 = giữ input, 0 = baseline
    baseline: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Up-sample patch_mask lên H×W và trộn với baseline.
    """
    B, C, H, W = x.shape
    _, _, Hp, Wp = patch_mask.shape
    if baseline is None:
        baseline = torch.zeros_like(x)

    mask_up = F.interpolate(
        patch_mask,
        size=(H, W),
        mode="nearest",
    )  # [B,1,H,W]

    return x * mask_up + baseline * (1.0 - mask_up)

def insertion_deletion_auc_single(
    model: torch.nn.Module,
    x: torch.Tensor,                # [1,3,H,W]
    patch_rel: torch.Tensor,        # [1,N_patches]
    target: torch.Tensor,           # [1], Long
    Hp: int,
    Wp: int,
    steps: int = 20,
    baseline: Optional[torch.Tensor] = None,
    use_logits: bool = True,
    eps: float = 1e-8,
) -> Tuple[float, float]:
    """
    Tính AUC cho insertion và deletion cho 1 ảnh.
    Trả về (auc_del, auc_ins).
    - deletion: bắt đầu từ ảnh gốc, dần dần che các patch quan trọng.
    - insertion: bắt đầu từ baseline, dần dần thêm các patch quan trọng.

    AUC tính trên score đã normalize về [0,1]:
      - deletion: chia cho score ban đầu.
      - insertion: chia cho score cuối.
    """
    device = x.device
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x)

    with torch.no_grad():
        logits = model(x)
        if use_logits:
            base_score = logits.gather(1, target.view(-1, 1)).squeeze(1)  # [1]
        else:
            probs = logits.softmax(dim=-1)
            base_score = probs.gather(1, target.view(-1, 1)).squeeze(1)   # [1]

    base_score = base_score.item()

    scores_del = []
    scores_ins = []

    Np = patch_rel.shape[1]
    # sort patch idx theo importance giảm dần
    importance = patch_rel[0].detach()
    idx_sorted = torch.argsort(importance, descending=True)  # [Np]

    step = max(Np // steps, 1)
    # số bước thực tế
    num_steps = (Np + step - 1) // step  # ceil

    # -------- Deletion --------
    for s in range(num_steps + 1):
        # che s*step patch quan trọng nhất
        k = s * step
        k = min(k, Np)
        # patch_mask = 1 ở patch giữ lại, 0 ở patch masked
        mask = torch.ones(1, 1, Hp, Wp, device=device)
        if k > 0:
            idx_del = idx_sorted[:k]
            h_del = idx_del // Wp
            w_del = idx_del % Wp
            mask[0, 0, h_del, w_del] = 0.0

        x_del = apply_patch_mask_to_image(x, mask, baseline=baseline)  # [1,3,H,W]
        with torch.no_grad():
            logits_del = model(x_del)
            if use_logits:
                score_del = logits_del.gather(1, target.view(-1, 1)).item()
            else:
                probs_del = logits_del.softmax(dim=-1)
                score_del = probs_del.gather(1, target.view(-1, 1)).item()
        scores_del.append(score_del)

    # normalize deletion scores theo base_score
    scores_del = np.array(scores_del, dtype=np.float64)
    if base_score > 0:
        scores_del_norm = np.clip(scores_del / (base_score + eps), 0.0, 1.0)
    else:
        # nếu base_score ~0, coi như không có signal, đặt toàn 0
        scores_del_norm = np.zeros_like(scores_del)
    xs_del = np.linspace(0.0, 1.0, num_steps + 1)
    auc_del = auc_trapezoid(xs_del, scores_del_norm)

    # -------- Insertion --------
    # bắt đầu từ ảnh baseline, dần thêm patch quan trọng
    scores_ins = []
    # để biết score cuối (tất cả patch), ta có thể dùng base_score luôn
    full_score = base_score

    for s in range(num_steps + 1):
        k = s * step
        k = min(k, Np)
        mask = torch.zeros(1, 1, Hp, Wp, device=device)
        if k > 0:
            idx_ins = idx_sorted[:k]
            h_ins = idx_ins // Wp
            w_ins = idx_ins % Wp
            mask[0, 0, h_ins, w_ins] = 1.0

        x_ins = apply_patch_mask_to_image(x, mask, baseline=baseline)
        with torch.no_grad():
            logits_ins = model(x_ins)
            if use_logits:
                score_ins = logits_ins.gather(1, target.view(-1, 1)).item()
            else:
                probs_ins = logits_ins.softmax(dim=-1)
                score_ins = probs_ins.gather(1, target.view(-1, 1)).item()
        scores_ins.append(score_ins)

    scores_ins = np.array(scores_ins, dtype=np.float64)
    if full_score > 0:
        scores_ins_norm = np.clip(scores_ins / (full_score + eps), 0.0, 1.0)
    else:
        scores_ins_norm = np.zeros_like(scores_ins)
    xs_ins = np.linspace(0.0, 1.0, num_steps + 1)
    auc_ins = auc_trapezoid(xs_ins, scores_ins_norm)

    return auc_del, auc_ins

def evaluate_xai_method_dataset(
    attributor,
    dataset,
    device: str = "cuda",
    use_pred: bool = True,
    max_samples: Optional[int] = None,
    steps: int = 20,
    baseline: Optional[torch.Tensor] = None,
    use_logits: bool = True,
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Đánh giá một XAI method trên dataset.

    attributor: object có .attribute(x, target, use_logits) -> (patch_rel, heatmap)
                và thuộc tính .model
    dataset: torch Dataset trả (img, label, ...) hoặc (img, label)

    Trả về dict:
        {
          "auc_del_mean": ...,
          "auc_ins_mean": ...,
          "entropy_mean": ...,
          "gini_mean": ...,
          "n_samples": N
        }
    """
    model = attributor.model
    model.eval()
    Hp, Wp = get_patch_grid(model)

    auc_del_list = []
    auc_ins_list = []
    ent_list = []
    gini_list = []

    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    for idx in range(n_total):
        sample = dataset[idx]
        if isinstance(sample, (tuple, list)):
            img = sample[0]
            label = sample[1]
        else:
            img = sample
            label = None

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 4:
            img = img[0]
        img = img.to(device).unsqueeze(0)  # [1,3,H,W]

        if isinstance(label, torch.Tensor):
            label = int(label.item())
        elif label is not None:
            label = int(label)

        # quyết định target class
        if use_pred or label is None:
            with torch.no_grad():
                logits = model(img)
                target_cls = int(logits.argmax(dim=-1).item())
        else:
            target_cls = label

        target_tensor = torch.tensor([target_cls], device=device, dtype=torch.long)

        # gọi attributor
        patch_rel, _ = attributor.attribute(img, target=target_tensor, use_logits=use_logits)
        # patch_rel: [1,N_p]
        patch_rel_1d = patch_rel[0]

        # sparsity metrics
        ent, gini = compute_entropy_and_gini(patch_rel_1d)
        ent_list.append(ent)
        gini_list.append(gini)

        # insertion/deletion AUC
        auc_del, auc_ins = insertion_deletion_auc_single(
            model=model,
            x=img,
            patch_rel=patch_rel,
            target=target_tensor,
            Hp=Hp,
            Wp=Wp,
            steps=steps,
            baseline=baseline,
            use_logits=use_logits,
        )
        auc_del_list.append(auc_del)
        auc_ins_list.append(auc_ins)

        if verbose_every and (idx + 1) % verbose_every == 0:
            print(
                f"[{idx+1}/{n_total}] "
                f"AUC_del_mean={np.mean(auc_del_list):.4f}, "
                f"AUC_ins_mean={np.mean(auc_ins_list):.4f}, "
                f"entropy_mean={np.mean(ent_list):.4f}"
            )

    result = {
        "auc_del_mean": float(np.mean(auc_del_list)) if auc_del_list else 0.0,
        "auc_ins_mean": float(np.mean(auc_ins_list)) if auc_ins_list else 0.0,
        "entropy_mean": float(np.mean(ent_list)) if ent_list else 0.0,
        "gini_mean": float(np.mean(gini_list)) if gini_list else 0.0,
        "n_samples": len(auc_del_list),
    }
    return result

def cosine_similarity_heatmaps(
    hm1: torch.Tensor,   # [1,1,H,W]
    hm2: torch.Tensor,   # [1,1,H,W]
    eps: float = 1e-8,
) -> float:
    v1 = hm1.view(-1).float()
    v2 = hm2.view(-1).float()

    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()

    n1 = torch.norm(v1)
    n2 = torch.norm(v2)
    if n1 < eps or n2 < eps:
        return 0.0

    cos = torch.dot(v1, v2) / (n1 * n2 + eps)
    return float(cos.item())

def evaluate_class_specificity_cosine(
    attributor,
    dataset,
    device: str = "cuda",
    max_samples: Optional[int] = None,
    use_logits: bool = True,
    use_pred: bool = True,
    verbose_every: int = 50,
) -> Dict[str, Any]:
    """
    Đánh giá class-specificity:
      - Với mỗi sample, lấy heatmap cho positive class (label hoặc predicted).
      - Lấy heatmap cho negative class (top-2 logit khác positive).
      - Tính cosine similarity giữa 2 heatmaps (flatten).
    Trả về:
      {
        "cosine_cs_mean": ...,
        "cosine_cs_std": ...,
        "n_samples": N
      }
    """
    model = attributor.model
    model.eval()

    cos_list = []

    n_total = len(dataset)
    if max_samples is not None:
        n_total = min(n_total, max_samples)

    for idx in range(n_total):
        sample = dataset[idx]
        if isinstance(sample, (tuple, list)):
            img = sample[0]
            label = sample[1]
        else:
            img = sample
            label = None

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        if img.ndim == 4:
            img = img[0]
        img = img.to(device).unsqueeze(0)  # [1,3,H,W]

        if isinstance(label, torch.Tensor):
            label = int(label.item())
        elif label is not None:
            label = int(label)

        # forward để lấy logits
        with torch.no_grad():
            logits = model(img)
            probs = logits.softmax(dim=-1)
            preds_sorted = torch.argsort(probs, dim=-1, descending=True)  # [1,num_classes]

        # quyết định positive class
        if use_pred or label is None:
            pos_cls = int(preds_sorted[0, 0].item())
        else:
            pos_cls = label

        # chọn negative class: lớp có prob cao nhưng ≠ pos_cls
        neg_cls = None
        for j in range(preds_sorted.shape[1]):
            c = int(preds_sorted[0, j].item())
            if c != pos_cls:
                neg_cls = c
                break
        if neg_cls is None:
            # trường hợp hiếm (1 class), bỏ qua
            continue

        pos_t = torch.tensor([pos_cls], device=device, dtype=torch.long)
        neg_t = torch.tensor([neg_cls], device=device, dtype=torch.long)

        # Heatmap cho positive class
        _, hm_pos = attributor.attribute(img, target=pos_t, use_logits=use_logits)
        # Heatmap cho negative class
        _, hm_neg = attributor.attribute(img, target=neg_t, use_logits=use_logits)

        # [1,1,H,W] -> tính cosine
        cos_val = cosine_similarity_heatmaps(hm_pos[0:1], hm_neg[0:1])
        cos_list.append(cos_val)

        if verbose_every and (idx + 1) % verbose_every == 0:
            print(f"[{idx+1}/{n_total}] cosine_cs_mean={np.mean(cos_list):.4f}")

    if len(cos_list) == 0:
        return {"cosine_cs_mean": 0.0, "cosine_cs_std": 0.0, "n_samples": 0}

    cos_arr = np.array(cos_list, dtype=np.float64)
    result = {
        "cosine_cs_mean": float(cos_arr.mean()),
        "cosine_cs_std": float(cos_arr.std()),
        "n_samples": len(cos_list),
    }
    return result

