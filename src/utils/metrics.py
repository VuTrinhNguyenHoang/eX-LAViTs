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

def tokens_to_heat(
    rtokens: torch.Tensor,
    model,
    x_shape,
    has_cls: bool = True
):
    B, C_in, H, W = x_shape
    assert B == rtokens.size(0)
    r = rtokens  # [B,N]

    # bỏ CLS nếu có
    if has_cls:
        r_patch = r[:, 1:]     # [B, Np]
    else:
        r_patch = r

    Np = r_patch.size(1)

    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        gh, gw = model.patch_embed.grid_size
    else:
        # fallback: suy từ stride/patch size
        if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
            S = model.patch_embed.proj.stride[0]
        else:
            raise RuntimeError("Không tìm được grid_size/stride từ model.")
        gh = H // S
        gw = W // S

    assert gh * gw == Np, f"grid_size ({gh},{gw}) không khớp với số patch {Np}"

    r_grid = r_patch.view(B, 1, gh, gw)  # [B,1,gh,gw]
    heat = F.interpolate(r_grid, size=(H, W), mode="bilinear", align_corners=False)  # [B,1,H,W]
    return heat[:, 0]  # [B,H,W]

def extract_heat_from_attr_output(
    attr_out: Dict[str, torch.Tensor],
    attr_model,
    x: torch.Tensor,
    has_cls: bool = True
):
    if "rimg" in attr_out:
        rimg = attr_out["rimg"]
        if rimg.dim() != 4:
            raise ValueError(f"rimg phải có shape [B,C,H,W], hiện tại: {rimg.shape}")
        heat = rimg.sum(dim=1)
        return heat
    
    rtokens = attr_out["rtokens"]
    model = getattr(attr_model, "model", None)
    if model is None:
        raise RuntimeError("Không tìm thấy attr_model.model để map rtokens -> heat.")
    return tokens_to_heat(rtokens, model, x.shape, has_cls=has_cls)

class AttributionMetrics:
    def __init__(self, model: nn.Module, device: str = "cuda", eps: float = 1e-6):
        self.model = model.to(device)
        self.device = device
        self.eps = eps

    # ------------- utils -------------
    def _prepare(self, x: torch.Tensor, heat: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:   # [3,H,W]
            x = x.unsqueeze(0)
        if heat.dim() == 2:  # [H,W]
            heat = heat.unsqueeze(0)
        assert x.dim() == 4 and heat.dim() == 3
        assert x.size(0) == heat.size(0)
        H, W = x.shape[-2:]
        assert heat.shape[-2:] == (H, W)
        x = x.to(self.device)
        heat = heat.to(self.device)
        return x, heat

    def _normalize_heat(self, heat: torch.Tensor) -> torch.Tensor:
        h = heat.clamp_min(0)
        s = h.flatten(1).sum(dim=1, keepdim=True) + self.eps
        h = h / s.view(-1, 1, 1)
        return h

    def _baseline(self, x: torch.Tensor, mode: str = "blur") -> torch.Tensor:
        if mode == "zeros":
            return torch.zeros_like(x)
        elif mode == "mean":
            m = x.mean(dim=(2,3), keepdim=True)
            return m.expand_as(x)
        elif mode == "blur":
            k = 11
            pad = k // 2
            return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)
        else:
            raise ValueError(f"Unknown baseline mode: {mode}")

    def _auc_trapezoid(self, xs: torch.Tensor, ys: torch.Tensor) -> float:
        xs = xs.view(-1)
        ys = ys.view(-1)
        dx = xs[1:] - xs[:-1]
        avg_y = 0.5 * (ys[1:] + ys[:-1])
        auc = (dx * avg_y).sum().item()
        return float(auc)

    # ------------- AUC Deletion & Insertion -------------
    @torch.no_grad()
    def auc_deletion(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        heat: torch.Tensor,
        steps: int = 20,
        baseline_mode: str = "mean",
    ) -> float:
        """
        AUC_deletion:
          - Bắt đầu từ ảnh gốc.
          - Lần lượt "xóa" (mask) các pixel từ cao -> thấp theo heatmap.
          - Theo dõi xác suất dự đoán class y_true.
          - AUC (cao -> ít sụt nhanh; thấp -> sụt nhanh, method tốt).
        """
        self.model.eval()
        x, heat = self._prepare(x, heat)   # [1,3,H,W], [1,H,W]
        y_true = y_true.view(-1).to(self.device)
        B, _, H, W = x.shape
        assert B == 1, "Hiện tại chỉ hỗ trợ batch=1 cho AUC_del/ins."

        # chuẩn hóa heat để xếp thứ tự
        h = heat.view(1, -1)  # [1, H*W]
        order = torch.argsort(h, dim=1, descending=True)  # [1, H*W]

        x_cur = x.clone()
        base = self._baseline(x, mode=baseline_mode)

        # mask ban đầu: all ones (không xóa)
        mask = torch.ones(1, 1, H*W, device=self.device)
        probs = []

        # bước = số phần tử xóa mỗi step
        num_pixels = H * W
        step = max(1, num_pixels // steps)

        # t=0: ảnh gốc
        with torch.no_grad():
            logits = self.model(x_cur)
            prob = logits.softmax(dim=1).gather(1, y_true[:, None]).squeeze(1)
        probs.append(prob.item())

        for i in range(1, steps+1):
            end = min(i * step, num_pixels)
            idx = order[:, :end].unsqueeze(1)  # [1, k]
            # cập nhật mask: 0 cho pixel đã xóa
            mask.scatter_(2, idx, 0.0)

            mask_2d = mask.view(1, 1, H, W)
            x_cur = x * mask_2d + base * (1 - mask_2d)

            logits = self.model(x_cur)
            prob = logits.softmax(dim=1).gather(1, y_true[:, None]).squeeze(1)
            probs.append(prob.item())

        xs = torch.linspace(0, 1, steps+1, device=self.device)  # tỉ lệ pixel đã xóa
        ys = torch.tensor(probs, device=self.device)
        auc = self._auc_trapezoid(xs, ys)
        return auc

    @torch.no_grad()
    def auc_insertion(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        heat: torch.Tensor,
        steps: int = 20,
        baseline_mode: str = "zeros",
    ) -> float:
        """
        AUC_insertion:
          - Bắt đầu từ baseline (ví dụ ảnh 0 hoặc blur).
          - Lần lượt "chèn" các pixel quan trọng nhất từ heatmap.
          - AUC (cao -> tăng nhanh về xác suất, method tốt).
        """
        self.model.eval()
        x, heat = self._prepare(x, heat)   # [1,3,H,W], [1,H,W]
        y_true = y_true.view(-1).to(self.device)
        B, _, H, W = x.shape
        assert B == 1

        h = heat.view(1, -1)                           # [1, H*W]
        order = torch.argsort(h, dim=1, descending=True)  # [1, H*W]

        base = self._baseline(x, mode=baseline_mode)
        x_cur = base.clone()

        # mask ban đầu: all zeros (chưa chèn)
        mask = torch.zeros(1, 1, H*W, device=self.device)
        probs = []

        num_pixels = H * W
        step = max(1, num_pixels // steps)

        # t=0: baseline
        logits = self.model(x_cur)
        prob = logits.softmax(dim=1).gather(1, y_true[:, None]).squeeze(1)
        probs.append(prob.item())

        for i in range(1, steps+1):
            end = min(i * step, num_pixels)
            idx = order[:, :end].unsqueeze(1)  # [1, k]
            # các pixel này được copy từ x sang
            mask.scatter_(2, idx, 1.0)

            mask_2d = mask.view(1, 1, H, W)
            x_cur = x * mask_2d + base * (1 - mask_2d)

            logits = self.model(x_cur)
            prob = logits.softmax(dim=1).gather(1, y_true[:, None]).squeeze(1)
            probs.append(prob.item())

        xs = torch.linspace(0, 1, steps+1, device=self.device)  # tỉ lệ pixel đã chèn
        ys = torch.tensor(probs, device=self.device)
        auc = self._auc_trapezoid(xs, ys)
        return auc

    # ------------- Entropy & Gini -------------
    def entropy(
        self,
        heat: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Entropy của phân bố relevance.
        heat: [B,H,W] hoặc [H,W] hoặc [B,N].
        Trả về: [B] tensor.
        """
        if heat.dim() == 2:      # [H,W] -> [1,H,W]
            heat = heat.unsqueeze(0)
        if heat.dim() == 3:      # [B,H,W] -> [B,HW]
            h = heat.flatten(1)
        elif heat.dim() == 2:    # [B,N]
            h = heat
        else:
            raise ValueError("heat shape không hỗ trợ")

        h = h.clamp_min(0)
        s = h.sum(dim=1, keepdim=True) + self.eps
        p = h / s

        ent = -(p * (p + self.eps).log()).sum(dim=1)  # [B]

        if normalize:
            K = p.size(1)
            ent = ent / (torch.log(torch.tensor(K, dtype=ent.dtype, device=ent.device)) + self.eps)
        return ent  # [B], trong [0,1] nếu normalize

    def gini_index(
        self,
        heat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gini index (dạng 'concentration'):
           G = sum_i p_i^2  (cao khi tập trung vào ít vùng, thấp khi phân tán).
        Nếu bạn muốn Gini impurity: 1 - G.
        """
        if heat.dim() == 2:
            heat = heat.unsqueeze(0)
        if heat.dim() == 3:
            h = heat.flatten(1)
        elif heat.dim() == 2:
            h = heat
        else:
            raise ValueError("heat shape không hỗ trợ")

        h = h.clamp_min(0)
        s = h.sum(dim=1, keepdim=True) + self.eps
        p = h / s
        g = (p ** 2).sum(dim=1)   # [B], 1/K <= g <= 1
        return g

    # ------------- Cross-class similarity -------------
    def cross_class_similarity(
        self,
        heat_multi: torch.Tensor,
        mode: str = "pairwise_mean",
    ) -> torch.Tensor:
        """
        Đo độ giống nhau giữa heatmap các class khác nhau cho cùng 1 ảnh.
        heat_multi: [C, H, W] hoặc [C, N] (C: số class giải thích).
        mode:
          - "pairwise_mean": mean cosine similarity của tất cả cặp (c1,c2), c1<c2.
          - "one_vs_rest": cosine giữa class 0 và các class còn lại, rồi lấy mean.
        Trả về: scalar tensor.
        """
        if heat_multi.dim() == 3:    # [C,H,W] -> [C,N]
            C, H, W = heat_multi.shape
            Hf = heat_multi.view(C, -1)
        elif heat_multi.dim() == 2:  # [C,N]
            Hf = heat_multi
            C = Hf.size(0)
        else:
            raise ValueError("heat_multi phải có shape [C,H,W] hoặc [C,N].")

        # chuẩn hóa dương + L2
        Hf = Hf.clamp_min(0)
        s = Hf.sum(dim=1, keepdim=True) + self.eps
        Hf = Hf / s

        # L2-normalize
        Hf = Hf / (Hf.norm(dim=1, keepdim=True) + self.eps)  # [C,N]

        if mode == "pairwise_mean":
            sims = []
            for i in range(C):
                for j in range(i+1, C):
                    sim = (Hf[i] * Hf[j]).sum()   # cosine
                    sims.append(sim)
            if len(sims) == 0:
                return torch.tensor(1.0, dtype=Hf.dtype, device=Hf.device)
            return torch.stack(sims).mean()
        elif mode == "one_vs_rest":
            base = Hf[0]    # class 0 vs others
            sims = []
            for j in range(1, C):
                sim = (base * Hf[j]).sum()
                sims.append(sim)
            if len(sims) == 0:
                return torch.tensor(1.0, dtype=Hf.dtype, device=Hf.device)
            return torch.stack(sims).mean()
        else:
            raise ValueError(f"Unknown mode {mode}")

def _get_item(sample):
        img, y = None, None
        if isinstance(sample, (tuple, list)):
            img = sample[0]
            if len(sample) > 1:
                y = sample[1]
        elif isinstance(sample, dict):
            for k in ("image", "img", "tensor", "x"):
                if k in sample:
                    img = sample[k]; break
            for k in ("label", "y", "target"):
                if k in sample:
                    y = sample[k]; break
        else:
            img = sample
        if not torch.is_tensor(img):
            raise TypeError("Dataset phải trả về ảnh dạng torch.Tensor.")
        if y is not None and not torch.is_tensor(y):
            y = torch.tensor(y)
        return img, y

def _mean(xs):
    if len(xs) == 0:
        return float("nan")
    return float(sum(xs) / len(xs))

def evaluate_method_on_loader(
    attr_model,
    backbone_model,
    dataloader,
    device: str = "cuda",
    has_cls: bool = True,
    steps: int = 20,
    max_batches: int = None
):
    metric = AttributionMetrics(backbone_model, device=device)
    attr_model.model.to(device)
    backbone_model.to(device)
    backbone_model.eval()

    auc_del_list = []
    auc_ins_list = []
    ent_list = []
    gini_list = []

    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating..."):
        if max_batches is not None and bi >= max_batches:
            break

        x, y = _get_item(batch)     # x: [B,3,H,W] hoặc [3,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if y is None:
            # nếu không có label GT, dùng predicted class
            with torch.no_grad():
                logits = backbone_model(x.to(device))
                y = logits.argmax(dim=1).cpu()
        else:
            y = y.view(-1)

        x = x.to(device)
        y = y.to(device)

        B = x.size(0)
        for i in range(B):
            xi = x[i:i+1]
            yi = y[i:i+1]

            attr_out = attr_model.attribute(xi, yi)
            heat = extract_heat_from_attr_output(attr_out, attr_model, xi, has_cls=has_cls)  # [1,H,W]
            heat_i = heat[0]

            auc_del = metric.auc_deletion(xi[0], yi[0], heat_i, steps=steps, baseline_mode="mean")
            auc_ins = metric.auc_insertion(xi[0], yi[0], heat_i, steps=steps, baseline_mode="zeros")

            ent = metric.entropy(heat_i, normalize=True)[0].item()
            gini = metric.gini_index(heat_i)[0].item()

            auc_del_list.append(auc_del)
            auc_ins_list.append(auc_ins)
            ent_list.append(ent)
            gini_list.append(gini)

    return {
        "auc_del_mean": _mean(auc_del_list),
        "auc_ins_mean": _mean(auc_ins_list),
        "entropy_mean": _mean(ent_list),
        "gini_mean": _mean(gini_list),
        "n_samples": len(auc_del_list),
    }

def evaluate_cross_class_similarity(
    attr_model,
    backbone_model,
    dataloader,
    device: str = "cuda",
    has_cls: bool = True,
    max_batches: int = None
):
    metric = AttributionMetrics(backbone_model, device=device)
    attr_model.model.to(device)
    backbone_model.to(device)
    backbone_model.eval()

    sims = []

    for bi, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating..."):
        if max_batches is not None and bi >= max_batches:
            break

        x, _ = _get_item(batch)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.to(device)

        B = x.size(0)
        for i in range(B):
            xi = x[i:i+1]   # [1,3,H,W]

            # top-2 class
            logits = backbone_model(xi)
            top2 = logits[0].topk(2).indices    # [2]
            classes = top2.to(device)

            heats = []
            for c in classes:
                yi = c.view(1)
                attr_out = attr_model.attribute(xi, yi)
                heat = extract_heat_from_attr_output(attr_out, attr_model, xi, has_cls=has_cls)  # [1,H,W]
                heats.append(heat[0])

            if len(heats) < 2:
                continue

            heat_multi = torch.stack(heats, dim=0)  # [2,H,W]
            sim = metric.cross_class_similarity(heat_multi, mode="pairwise_mean")
            sims.append(sim.item())

    if len(sims) == 0:
        return {"ccs_mean": float("nan"), "n_samples": 0}
    return {"ccs_mean": float(sum(sims)/len(sims)), "n_samples": len(sims)}

