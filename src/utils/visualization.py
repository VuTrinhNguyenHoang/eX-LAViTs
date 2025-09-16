import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
import os
from sklearn.metrics import confusion_matrix
import torch

from .metrics import MetricsTracker

def plot_training_curves(
    metrics_tracker: MetricsTracker,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        metrics_tracker: MetricsTracker containing training history
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    epochs = range(1, len(metrics_tracker.train_losses) + 1)
    
    # Plot loss curves
    axes[0].plot(epochs, metrics_tracker.train_losses, 'b-', label='Train Loss', linewidth=2)
    if metrics_tracker.val_losses:
        axes[0].plot(epochs, metrics_tracker.val_losses, 'r-', label='Val Loss', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy curves
    axes[1].plot(epochs, metrics_tracker.train_accuracies, 'b-', label='Train Acc', linewidth=2)
    if metrics_tracker.val_accuracies:
        axes[1].plot(epochs, metrics_tracker.val_accuracies, 'r-', label='Val Acc', linewidth=2)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

def plot_learning_rate_schedule(
    metrics_tracker: MetricsTracker,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 4),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot learning rate schedule.
    
    Args:
        metrics_tracker: MetricsTracker containing learning rates
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        matplotlib Figure object
    """
    if not metrics_tracker.learning_rates:
        raise ValueError("No learning rate data available")
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    epochs = range(1, len(metrics_tracker.learning_rates) + 1)
    ax.plot(epochs, metrics_tracker.learning_rates, 'g-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    dpi: int = 100,
    cmap: str = 'Blues'
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        cmap: Colormap for the plot
        
    Returns:
        matplotlib Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

def plot_metrics_comparison(
    metrics_dict: Dict[str, List[float]],
    metric_name: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot comparison of metrics across different models/runs.
    
    Args:
        metrics_dict: Dictionary with keys as model names and values as metric lists
        metric_name: Name of the metric being plotted
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    for model_name, values in metrics_dict.items():
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, label=model_name, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

def plot_attention_weights(
    attention_weights: torch.Tensor,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = 100,
    head_idx: int = 0,
    layer_idx: int = -1
) -> plt.Figure:
    """
    Plot attention weights heatmap.
    
    Args:
        attention_weights: Attention weights tensor [batch, heads, seq_len, seq_len]
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        head_idx: Index of attention head to visualize
        layer_idx: Index of layer to visualize
        
    Returns:
        matplotlib Figure object
    """
    # Take first sample in batch and specified head
    attn = attention_weights[0, head_idx].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    sns.heatmap(
        attn,
        cmap='viridis',
        square=True,
        cbar_kws={'label': 'Attention Weight'},
        ax=ax
    )
    
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

def plot_model_comparison_bar(
    model_results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'loss'],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 100
) -> plt.Figure:
    """
    Plot bar chart comparing models across different metrics.
    
    Args:
        model_results: Dictionary with model names as keys and metric dictionaries as values
        metrics: List of metrics to plot
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        matplotlib Figure object
    """
    df_data = []
    for model_name, results in model_results.items():
        for metric in metrics:
            if metric in results:
                df_data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': results[metric]
                })
    
    df = pd.DataFrame(df_data)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    sns.barplot(data=df, x='Model', y='Value', hue='Metric', ax=ax)
    
    ax.set_title('Model Performance Comparison')
    ax.set_ylabel('Value')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig

def save_all_plots(
    metrics_tracker: MetricsTracker,
    run_dir: str,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    dpi: int = 100
) -> None:
    """
    Save all training plots to run directory.
    
    Args:
        metrics_tracker: MetricsTracker containing training history
        run_dir: Run directory to save plots
        y_true: True labels for confusion matrix (optional)
        y_pred: Predicted labels for confusion matrix (optional)
        class_names: List of class names (optional)
        dpi: Figure DPI
    """
    plots_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training curves
    fig = plot_training_curves(metrics_tracker, dpi=dpi)
    fig.savefig(os.path.join(plots_dir, 'training_curves.png'), bbox_inches='tight', dpi=dpi)
    plt.close(fig)
    
    # Learning rate schedule
    if metrics_tracker.learning_rates:
        fig = plot_learning_rate_schedule(metrics_tracker, dpi=dpi)
        fig.savefig(os.path.join(plots_dir, 'learning_rate.png'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)
    
    # Confusion matrix
    if y_true is not None and y_pred is not None:
        # Regular confusion matrix
        fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, dpi=dpi)
        fig.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)
        
        # Normalized confusion matrix
        fig = plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, dpi=dpi)
        fig.savefig(os.path.join(plots_dir, 'confusion_matrix_normalized.png'), bbox_inches='tight', dpi=dpi)
        plt.close(fig)

def create_training_summary_plot(
    metrics_tracker: MetricsTracker,
    model_info: Dict[str, Any],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 10),
    dpi: int = 100
) -> plt.Figure:
    """
    Create a comprehensive training summary plot.
    
    Args:
        metrics_tracker: MetricsTracker containing training history
        model_info: Dictionary containing model information
        save_path: Path to save plot (optional)
        figsize: Figure size
        dpi: Figure DPI
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize, dpi=dpi)
    
    # Create subplots - now with 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    epochs = range(1, len(metrics_tracker.train_losses) + 1)
    
    # Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, metrics_tracker.train_losses, 'b-', label='Train', linewidth=2)
    if metrics_tracker.val_losses:
        ax1.plot(epochs, metrics_tracker.val_losses, 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, metrics_tracker.train_accuracies, 'b-', label='Train', linewidth=2)
    if metrics_tracker.val_accuracies:
        ax2.plot(epochs, metrics_tracker.val_accuracies, 'r-', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Learning rate
    if metrics_tracker.learning_rates:
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, metrics_tracker.learning_rates, 'g-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # Model information text (left side of bottom row)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    
    # Create summary text
    summary = metrics_tracker.get_summary()
    info_text = f"""Model Information:
• Model Type: {model_info.get('model_type', 'N/A')}
• Total Parameters: {model_info.get('total_params', 'N/A'):,} 
• Image Size: {model_info.get('img_size', 'N/A')}
• Patch Size: {model_info.get('patch_size', 'N/A')}
• Embed Dim: {model_info.get('embed_dim', 'N/A')}
• Num Heads: {model_info.get('num_heads', 'N/A')}
• Depth: {model_info.get('depth', 'N/A')}

Training Summary:
• Best Train Accuracy: {summary.get('best_train_accuracy', 0):.2f}%
• Best Val Accuracy: {summary.get('best_val_accuracy', 0):.2f}%
• Final Train Loss: {summary.get('final_train_loss', 0):.4f}
• Final Val Loss: {summary.get('final_val_loss', 0):.4f}
• Best Val Acc Epoch: {summary.get('best_val_accuracy_epoch', 0) + 1}"""
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Training time plot (bottom middle)
    if metrics_tracker.epoch_times:
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(epochs, metrics_tracker.epoch_times, 'purple', marker='o', linewidth=2, markersize=4)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Time (seconds)')
        ax5.set_title('Training Time per Epoch')
        ax5.grid(True, alpha=0.3)
        
        # Add average line
        avg_time = np.mean(metrics_tracker.epoch_times)
        ax5.axhline(y=avg_time, color='orange', linestyle='--', alpha=0.7, label=f'Avg: {avg_time:.1f}s')
        ax5.legend()
    
    # Memory usage plot (bottom right)
    if metrics_tracker.memory_usage:
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(epochs, metrics_tracker.memory_usage, 'red', marker='s', linewidth=2, markersize=4)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Memory Usage (MB)')
        ax6.set_title('Memory Usage per Epoch')
        ax6.grid(True, alpha=0.3)
        
        # Add peak line
        peak_memory = max(metrics_tracker.memory_usage)
        ax6.axhline(y=peak_memory, color='darkred', linestyle='--', alpha=0.7, label=f'Peak: {peak_memory:.0f}MB')
        ax6.legend()

    plt.suptitle('Training Summary Report', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    return fig