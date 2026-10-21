import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..models.explainability import (
    ViTGradCAM,
    IntegratedGradient,
    Rollout,
    LAGRA,
    TextGradCAM,
    TextIntegratedGradient,
    TextRollout,
    TextLAGRA,
)
from ..models.explainability.attention_baselines import (
    CDAM,
    GMAR,
    GradientRollout,
    set_record_attention,
)
from .metrics import (
    auc_trapezoid,
    compute_entropy_and_gini,
    cosine_similarity_heatmaps,
    get_patch_grid,
    insertion_deletion_auc_single,
)


def method_factories(model, text=False):
    return {
        "GradCAM": lambda: (TextGradCAM if text else ViTGradCAM)(model),
        "IG": lambda: (TextIntegratedGradient if text else IntegratedGradient)(
            model, m_steps=32
        ),
        "Rollout": lambda: (TextRollout if text else Rollout)(model),
        "GradientRollout": lambda: GradientRollout(model, text=text),
        "CDAM": lambda: CDAM(model, layer="last", text=text),
        "GMAR": lambda: GMAR(model, text=text),
        "Ours": lambda: (TextLAGRA if text else LAGRA)(model),
    }


def release_attributor(attributor):
    for name in ("handle_fwd", "handle_bwd"):
        handle = getattr(attributor, name, None)
        if handle is not None:
            handle.remove()
    for name in ("feats", "grads"):
        if hasattr(attributor, name):
            setattr(attributor, name, None)
    set_record_attention(attributor.model, False)
    attributor.model.zero_grad(set_to_none=True)


def insertion_deletion_auc_text(
    model, input_ids, token_rel, target, steps=20, baseline_id=0, use_logits=True, eps=1e-8
):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        values = logits if use_logits else logits.softmax(dim=-1)
        base_score = values.gather(1, target.reshape(-1, 1)).item()
    length = token_rel.shape[1]
    order = torch.argsort(token_rel[0].detach(), descending=True)
    step = max(length // steps, 1)
    num_steps = (length + step - 1) // step
    curves = []
    for insertion in (False, True):
        scores = []
        for index in range(num_steps + 1):
            count = min(index * step, length)
            ids = torch.full_like(input_ids, baseline_id) if insertion else input_ids.clone()
            ids[0, order[:count]] = input_ids[0, order[:count]] if insertion else baseline_id
            with torch.no_grad():
                logits = model(ids)
                values = logits if use_logits else logits.softmax(dim=-1)
                scores.append(values.gather(1, target.reshape(-1, 1)).item())
        scores = np.asarray(scores, dtype=np.float64)
        normalized = (
            np.clip(scores / (base_score + eps), 0.0, 1.0)
            if base_score > 0
            else np.zeros_like(scores)
        )
        curves.append(auc_trapezoid(np.linspace(0.0, 1.0, num_steps + 1), normalized))
    return tuple(curves)


def rationale_metrics(scores, rationale):
    rationale = np.asarray(rationale, dtype=np.int64).reshape(-1)
    count = int(rationale.sum())
    if count == 0:
        return {}
    scores = scores.detach().cpu().numpy().reshape(-1)
    predicted = np.zeros_like(scores, dtype=bool)
    predicted[np.argsort(-scores)[:count]] = True
    length = min(len(predicted), len(rationale))
    predicted, actual = predicted[:length], rationale[:length].astype(bool)
    tp = np.logical_and(predicted, actual).sum()
    fp = np.logical_and(predicted, ~actual).sum()
    fn = np.logical_and(~predicted, actual).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return {
        "rationale_prec": float(precision),
        "rationale_rec": float(recall),
        "rationale_f1": float(2 * precision * recall / (precision + recall + 1e-8)),
        "rationale_iou": float(tp / (np.logical_or(predicted, actual).sum() + 1e-8)),
    }


def summarize_samples(samples):
    frame = pd.DataFrame(samples)
    metrics = [
        name
        for name in (
            "auc_del",
            "auc_ins",
            "entropy",
            "gini",
            "cosine_cs",
            "rationale_prec",
            "rationale_rec",
            "rationale_f1",
            "rationale_iou",
        )
        if name in frame.columns
    ]
    rows = []
    for method, group in frame.groupby("method", sort=False):
        row = {"method": method, "n_samples": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            row.update(
                {
                    f"{metric}_mean": values.mean(),
                    f"{metric}_std": values.std(ddof=1),
                    f"{metric}_n": len(values),
                }
            )
        row["zero_map_count"] = int(group["zero_map"].sum())
        row["constant_cosine_count"] = int(group["constant_cosine_map"].sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def format_summary(summary):
    result = pd.DataFrame(index=summary.index)
    for name in summary.columns:
        if name.endswith("_mean"):
            metric = name[:-5]
            result[metric] = [
                f"{mean:.4f} ± {std:.4f}"
                for mean, std in zip(summary[name], summary[f"{metric}_std"])
            ]
    result["n_samples"] = summary["n_samples"]
    return result


def evaluate_methods(
    model,
    dataset,
    device,
    output_dir,
    max_samples=None,
    steps=20,
    pad_id=None,
    sample_ids=None,
    factories=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    text = pad_id is not None
    factories = method_factories(model, text=text) if factories is None else factories
    count = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    rows = []
    started = time.perf_counter()
    with (output_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for index in tqdm(range(count), desc="Evaluation", mininterval=20):
            sample = dataset[index]
            data, label = (sample["input_ids"], sample["label"]) if text else sample[:2]
            x = torch.as_tensor(data).unsqueeze(0).to(device)
            label = int(torch.as_tensor(label).item())
            with torch.no_grad():
                logits = model(x)
                target = logits.argmax(dim=-1)
                ranked = logits.softmax(dim=-1).argsort(dim=-1, descending=True)[0]
                negative = ranked[ranked != target.item()][:1]
            common = {
                "sample_index": index,
                "sample_id": str(sample_ids[index]) if sample_ids is not None else str(index),
                "input_sha256": hashlib.sha256(
                    x.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest(),
                "label": label,
                "target": int(target.item()),
                "negative_target": int(negative.item()),
                "correct": int(target.item()) == label,
                "base_logit": float(logits[0, target.item()].item()),
                "nonpositive_base_logit": bool(logits[0, target.item()].item() <= 0),
            }
            for name, factory in factories.items():
                attributor = factory()
                try:
                    relevance, heatmap = attributor.attribute(
                        x, target=target, use_logits=True
                    )
                    relevance, heatmap = relevance.detach(), heatmap.detach()
                    _, negative_map = attributor.attribute(x, target=negative, use_logits=True)
                    negative_map = negative_map.detach()
                finally:
                    release_attributor(attributor)
                entropy, gini = compute_entropy_and_gini(relevance[0])
                if text:
                    auc_del, auc_ins = insertion_deletion_auc_text(
                        model, x, relevance, target, steps=steps, baseline_id=pad_id
                    )
                else:
                    hp, wp = get_patch_grid(model)
                    auc_del, auc_ins = insertion_deletion_auc_single(
                        model, x, relevance, target, hp, wp, steps=steps, use_logits=True
                    )
                constant = any(
                    (value.float().flatten() - value.float().mean()).norm().item() < 1e-8
                    for value in (heatmap, negative_map)
                )
                row = dict(
                    common,
                    method=name,
                    auc_del=auc_del,
                    auc_ins=auc_ins,
                    entropy=entropy,
                    gini=gini,
                    cosine_cs=cosine_similarity_heatmaps(heatmap, negative_map),
                    constant_cosine_map=constant,
                    zero_map=bool(relevance.count_nonzero().item() == 0),
                )
                if text and "rationale_mask" in sample:
                    row.update(rationale_metrics(relevance[0], sample["rationale_mask"]))
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                rows.append(row)
            if (index + 1) % 10 == 0 or index + 1 == count:
                print(
                    f"Evaluated {index + 1}/{count} samples; {(time.perf_counter() - started) / 60:.1f} min",
                    flush=True,
                )
    summary = summarize_samples(rows)
    pd.DataFrame(rows).to_csv(output_dir / "samples.csv", index=False)
    summary.to_csv(output_dir / "summary.csv")
    format_summary(summary).to_csv(output_dir / "summary_formatted.csv")
    return summary


def save_metadata(output_dir, config, model, checkpoint_path=None, method_details=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata = {
        "config": config,
        "git_revision": revision,
        "python": platform.python_version(),
        "source_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn")
        },
        "device": str(next(model.parameters()).device),
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "protocol": {
            "target": "original predicted class; negative is highest-probability other class",
            "auc": "logit / (original target logit + 1e-8), clipped to [0,1]; zero curve if original logit <= 0",
            "grid": "legacy floor(N / steps) increments; linearly spaced integration axis",
            "baseline": "PAD token ID for text; zero in normalized image/spectrogram space otherwise",
            "text_positions": "all supplied positions including special tokens and padding; no padding attention mask",
            "cosine": "mean-centered heatmaps; constant map yields legacy value 0 and a separate flag",
            "std": "sample standard deviation across examples, ddof=1; not training-seed variability",
        },
        "methods": {
            "GradCAM": "original repository implementation at model.norm",
            "IG": "original repository implementation; 32 integration steps",
            "Rollout": "original repository implementation",
            "GradientRollout": "gradient-only adaptation: positive attention*gradient, head mean, add identity, row normalize, M_L...M_1; no LRP",
            "CDAM": "user-supplied last-layer attention-gradient implementation; not independently verified against the published method",
            "GMAR": "user-supplied head-weighted attention implementation; not independently verified against the published method",
            "Ours": "LAGRA/TextLAGRA; double normalization; CLS row-vector rollout; M_1...M_L",
        },
        "gradient_rollout_reference": {
            "paper": "Chefer et al., Transformer Interpretability Beyond Attention Visualization, CVPR 2021 [28]",
            "source": "https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_LRP.py",
            "scope": "uses raw reconstructed attention instead of LRP attention relevance; not the full method in [28]",
        },
    }
    if method_details is not None:
        metadata["methods"] = method_details
    if "GradientRollout" not in metadata["methods"]:
        metadata.pop("gradient_rollout_reference")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        with checkpoint_path.open("rb") as handle:
            metadata["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata["checkpoint_path"] = str(checkpoint_path)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    return metadata


def benchmark_methods(model, x, output_dir, text=False, warmup=3, repeats=10, factories=None):
    device = x.device
    model.eval()
    with torch.no_grad():
        target = model(x).argmax(dim=-1)
    factories = method_factories(model, text=text) if factories is None else factories
    factories = {"Forward": None, **factories}
    rows = []
    for name, factory in factories.items():
        attributor = factory() if factory is not None else None
        times, peaks, additional = [], [], []
        try:
            for index in range(warmup + repeats):
                model.zero_grad(set_to_none=True)
                if attributor is not None:
                    for attr in ("feats", "grads"):
                        if hasattr(attributor, attr):
                            setattr(attributor, attr, None)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    baseline = torch.cuda.memory_allocated(device)
                start = time.perf_counter()
                if attributor is None:
                    with torch.no_grad():
                        output = model(x)
                else:
                    output = attributor.attribute(x, target=target, use_logits=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - start) * 1000
                if index >= warmup:
                    times.append(elapsed)
                    if device.type == "cuda":
                        peak = torch.cuda.max_memory_allocated(device)
                        peaks.append(peak / 2**20)
                        additional.append((peak - baseline) / 2**20)
                del output
        finally:
            if attributor is not None:
                release_attributor(attributor)
        rows.append(
            {
                "method": name,
                "latency_ms_mean": np.mean(times),
                "latency_ms_median": np.median(times),
                "latency_ms_std": np.std(times, ddof=1),
                "peak_allocated_mib": max(peaks) if peaks else None,
                "additional_peak_mib": max(additional) if additional else None,
                "repeats": repeats,
                "warmup": warmup,
                "batch_size": len(x),
            }
        )
        print(f"{name}: {np.mean(times):.2f} ± {np.std(times, ddof=1):.2f} ms", flush=True)
    result = pd.DataFrame(rows).set_index("method")
    result.to_csv(Path(output_dir) / "compute.csv")
    return result
