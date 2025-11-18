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

def get_item_from_dataset(ds, idx):
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

def to_device_batch(x, y, device):
    x = x.unsqueeze(0).to(device, non_blocking=True)  # [1,3,H,W]
    y = None if y is None else y.view(-1).to(device)
    return x, y

def get_patch_grid_and_size(model: nn.Module, x: torch.Tensor) -> Tuple[int,int,int,int]:
    """
    Trả về (H_p, W_p, patch_h, patch_w)
    H_p,W_p: lưới patch
    patch_h,patch_w: kích thước mỗi patch trên ảnh
    """
    grid_hw = getattr(getattr(model, "patch_embed", None), "grid_size", None)
    B, C, H, W = x.shape
    if grid_hw is not None:
        H_p, W_p = grid_hw
    else:
        # fallback: assume square
        side = int((H * W) ** 0.5)
        H_p = W_p = H // 16
    patch_h = H // H_p
    patch_w = W // W_p
    return H_p, W_p, patch_h, patch_w

def apply_patches(
    base: torch.Tensor,
    src: torch.Tensor,
    patch_indices: torch.Tensor,
    model: nn.Module,
    mode: str = "insert",
):
    """
    base, src: [1,3,H,W]
    patch_indices: 1D tensor of patch ids (0..N_p-1)
    mode:
        - 'insert': copy patch từ src -> base
        - 'delete': set patch ở base = 0
    """
    assert base.shape == src.shape and base.shape[0] == 1
    device = base.device

    H_p, W_p, patch_h, patch_w = get_patch_grid_and_size(model, base)
    x_out = base.clone()

    for idx in patch_indices.tolist():
        i = idx // W_p
        j = idx % W_p
        h0, h1 = i * patch_h, (i + 1) * patch_h
        w0, w1 = j * patch_w, (j + 1) * patch_w

        if mode == "insert":
            x_out[:, :, h0:h1, w0:w1] = src[:, :, h0:h1, w0:w1]
        elif mode == "delete":
            x_out[:, :, h0:h1, w0:w1] = 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return x_out

def entropy_gini_from_tokens(token_scores: torch.Tensor, eps: float = 1e-6):
    """
    token_scores: [N] hoặc [1,N]; không cần chuẩn hoá trước.
    Trả về (entropy_norm, gini), đều là scalar float.
    """
    if token_scores.dim() == 2:
        token_scores = token_scores[0]
    # đảm bảo dương và chuẩn hoá thành phân phối
    scores = torch.clamp(token_scores, min=0)
    s_sum = scores.sum() + eps
    p = scores / s_sum      # [N]
    N = p.numel()

    # entropy chuẩn hoá 0..1
    entropy = -(p * (p + eps).log()).sum()
    entropy_norm = (entropy / math.log(N + eps)).item()

    # Gini với sum(p)=1: G = 1 - Σ p_i^2
    gini = (1.0 - (p * p).sum()).item()
    return entropy_norm, gini

def compute_del_ins_auc_single(
    explainer,
    x: torch.Tensor,
    target_cls: torch.Tensor,
    steps: Optional[int] = None,
    use_proba: bool = False,
    eps: float = 1e-6,
):
    """
    explainer: một trong các method (LAGR, GradCAM, ...)
    x: [1,3,H,W]
    target_cls: [1] (index class)
    steps: số bước, nếu None -> = N_patches
    use_proba: True -> dùng softmax prob; False -> dùng logit

    Trả về (auc_del, auc_ins)
    """
    model = explainer.model
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    target_cls = target_cls.to(device)

    # 1) lấy importance map
    with torch.no_grad():
        token_scores, _ = explainer.attribute(x, target=target_cls, img_size=None)
        # token_scores: [1,N_patches]
    scores = token_scores[0]  # [N]

    # ranking patch theo độ quan trọng giảm dần
    order = torch.argsort(scores, descending=True)  # [N]
    N = order.numel()
    if steps is None:
        steps = N
    step_size = max(1, N // steps)

    # 2) logit hoặc prob gốc
    with torch.no_grad():
        logits = model(x)  # [1,C]
    if use_proba:
        probs = logits.softmax(dim=-1)
        base_val = probs[0, target_cls.item()]
    else:
        base_val = logits[0, target_cls.item()]

    # baseline all-zero
    baseline = torch.zeros_like(x)

    # ------- Deletion: start from x, progressively x -> baseline -------
    x_del = x.clone()
    del_vals = []

    with torch.no_grad():
        for i in range(0, N + 1, step_size):
            logits_d = model(x_del)
            if use_proba:
                v = logits_d.softmax(dim=-1)[0, target_cls.item()]
            else:
                v = logits_d[0, target_cls.item()]
            del_vals.append(v.item())

            if i == N:
                break
            patch_ids = order[i : min(i + step_size, N)]
            x_del = apply_patches(x_del, baseline, patch_ids, model, mode="delete")

    del_vals = torch.tensor(del_vals, device=device)  # [T]
    # chuẩn hoá curve: start=1, end=0
    del_norm = (del_vals - del_vals[-1]) / (del_vals[0] - del_vals[-1] + eps)
    xs = torch.linspace(0, 1, del_norm.numel(), device=device)
    auc_del = torch.trapz(del_norm, xs).item()

    # ------- Insertion: start from baseline, progressively baseline -> x -------
    x_ins = baseline.clone()
    ins_vals = []

    with torch.no_grad():
        for i in range(0, N + 1, step_size):
            logits_i = model(x_ins)
            if use_proba:
                v = logits_i.softmax(dim=-1)[0, target_cls.item()]
            else:
                v = logits_i[0, target_cls.item()]
            ins_vals.append(v.item())

            if i == N:
                break
            patch_ids = order[i : min(i + step_size, N)]
            x_ins = apply_patches(x_ins, x, patch_ids, model, mode="insert")

    ins_vals = torch.tensor(ins_vals, device=device)
    ins_norm = (ins_vals - ins_vals[0]) / (ins_vals[-1] - ins_vals[0] + eps)
    xs = torch.linspace(0, 1, ins_norm.numel(), device=device)
    auc_ins = torch.trapz(ins_norm, xs).item()

    return auc_del, auc_ins

def compute_ccs_top2(
    explainer,
    x: torch.Tensor,
    logits: torch.Tensor,
) -> float:
    model = explainer.model
    model.eval()
    device = x.device

    probs, idxs = torch.topk(logits[0], k=min(2, logits.shape[1]))
    if idxs.numel() < 2:
        # chỉ 1 class, CCS không meaningful
        return float("nan")

    c1, c2 = idxs[0].view(1), idxs[1].view(1)  # [1]

    with torch.no_grad():
        _, h1 = explainer.attribute(x, target=c1, img_size=None)
        _, h2 = explainer.attribute(x, target=c2, img_size=None)

    h1 = h1[0, 0].reshape(-1)  # [HW]
    h2 = h2[0, 0].reshape(-1)

    # chuẩn hoá vector
    h1 = h1 - h1.mean()
    h2 = h2 - h2.mean()
    h1 = h1 / (h1.norm() + 1e-6)
    h2 = h2 / (h2.norm() + 1e-6)

    cos = float((h1 * h2).sum().item())
    return cos

def evaluate_methods_on_dataset(
    methods: Dict[str, nn.Module],
    dataset,
    device: str,
    use_pred: bool = True,
    max_samples: Optional[int] = None,
    steps_auc: Optional[int] = None,
    use_proba_auc: bool = False,
    compute_ccs: bool = True,
):
    """
    methods: tên -> explainer (LAGR, GradCAM, IG, KernelSHAP, Occlusion, Rollout)
    dataset: test_dataset
    use_pred:
        True  -> target = class dự đoán
        False -> nếu dataset có label thì dùng label
    """
    device_t = torch.device(device)

    first_explainer = next(iter(methods.values()))
    backbone = first_explainer.model.to(device_t).eval()

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    metrics = {
        name: defaultdict(list)
        for name in methods.keys()
    }

    for idx in tqdm(range(n), desc="Evaluating XAI methods"):
        img_t, y = get_item_from_dataset(dataset, idx)
        x, y = to_device_batch(img_t, y, device_t)

        with torch.no_grad():
            logits = backbone(x)          # [1,C]
            pred_cls = logits.argmax(dim=1)

        target_cls = pred_cls if (use_pred or y is None) else y

        for name, explainer in methods.items():
            explainer.model.to(device_t).eval()

            # 1) token_scores cho entropy/gini + AUC
            with torch.no_grad():
                token_scores, _ = explainer.attribute(
                    x, target=target_cls, img_size=None
                )  # [1,N], [1,1,H,W]

            entropy, gini = entropy_gini_from_tokens(token_scores[0])

            auc_del, auc_ins = compute_del_ins_auc_single(
                model=explainer.model,
                x=x,
                target_cls=target_cls,
                token_scores=token_scores,
                steps=steps_auc,
                use_proba=use_proba_auc,
            )

            metrics[name]["auc_del"].append(auc_del)
            metrics[name]["auc_ins"].append(auc_ins)
            metrics[name]["entropy"].append(entropy)
            metrics[name]["gini"].append(gini)

            # 2) CCS: cosine giữa heatmap class1 và class2
            if compute_ccs:
                ccs = compute_ccs_top2(explainer, x, logits)
                if not math.isnan(ccs):
                    metrics[name]["ccs"].append(ccs)

    # tổng hợp
    out = {}
    for name, vals in metrics.items():
        m = {}
        if len(vals["auc_del"]) > 0:
            m["auc_del_mean"] = float(sum(vals["auc_del"]) / len(vals["auc_del"]))
            m["auc_ins_mean"] = float(sum(vals["auc_ins"]) / len(vals["auc_ins"]))
            m["entropy_mean"] = float(sum(vals["entropy"]) / len(vals["entropy"]))
            m["gini_mean"] = float(sum(vals["gini"]) / len(vals["gini"]))
            if compute_ccs and len(vals["ccs"]) > 0:
                m["ccs_mean"] = float(sum(vals["ccs"]) / len(vals["ccs"]))
            else:
                m["ccs_mean"] = None
            m["n_samples"] = len(vals["auc_del"])
        out[name] = m

    return out
