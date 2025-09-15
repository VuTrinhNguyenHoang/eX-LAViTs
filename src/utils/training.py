import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time
from datetime import datetime
import yaml

from .metrics import AverageMeter, accuracy, compute_loss_and_accuracy, MetricsTracker

def create_run_directory(base_dir: str = "experiments", run_name: Optional[str] = "test") -> str:
    """
    Create a new run directory with incremental naming.
    
    Args:
        base_dir: Base directory for experiments
        run_name: Optional custom run name. Uses run001, run002, etc.
        
    Returns:
        Path to the created run directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "configs"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    
    return run_dir

def setup_logging(log_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        log_dir: Directory to save log file
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    log_file = os.path.join(log_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup logger
    logger = logging.getLogger('training')
    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    save_path: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Dictionary of metrics
        save_path: Path to save checkpoint
        is_best: Whether this is the best checkpoint so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: Learning rate scheduler (optional)
        device: Device to load checkpoint to
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> optim.Optimizer:
    """
    Get optimizer for training.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'adamw', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, **kwargs)
    elif optimizer_name.lower() == 'sgd':
        momentum = kwargs.pop('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    **kwargs
) -> Optional[Any]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler ('cosine', 'step', 'plateau', 'none')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler or None
    """
    if scheduler_name.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name.lower() == 'step':
        step_size = kwargs.get('step_size', 30)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name.lower() == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.1)
        patience = kwargs.get('patience', 10)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_name.lower() == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency
        logger: Logger for output
        
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images, target = images.to(device), target.to(device)
        
        # Compute output
        output = model(images)
        loss = criterion(output, target)
        
        # Measure record loss
        losses.update(loss.item(), images.size(0))
        
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            msg = (f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                   f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   f'Loss {losses.val:.4f} ({losses.avg:.4f})\t')
            
            if logger:
                logger.info(msg)
            else:
                print(msg)
    
    return {
        'loss': losses.avg,
        'batch_time': batch_time.avg
    }

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        logger: Logger for output
        
    Returns:
        Dictionary containing validation metrics
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # Switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            
            # Compute output
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
    msg = (f'Validation: [{epoch}]\t'
           f'Time {batch_time.avg:.3f}\t'
           f'Loss {losses.avg:.4f}\t'
           f'Acc@1 {top1.avg:.3f}\t'
           f'Acc@5 {top5.avg:.3f}')
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return {
        'loss': losses.avg,
        'top1_accuracy': top1.avg,
        'top5_accuracy': top5.avg,
        'batch_time': batch_time.avg
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    num_epochs: int,
    run_dir: str,
    save_freq: int = 10,
    print_freq: int = 50,
    early_stopping_patience: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> MetricsTracker:
    """
    Complete training loop.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device to train on
        num_epochs: Number of epochs to train
        run_dir: Directory to save checkpoints and logs
        save_freq: Frequency to save checkpoints
        print_freq: Print frequency during training
        early_stopping_patience: Early stopping patience (optional)
        logger: Logger for output
        
    Returns:
        MetricsTracker containing training history
    """
    metrics_tracker = MetricsTracker()
    best_val_acc = 0.0
    early_stopping_counter = 0
    
    models_dir = os.path.join(run_dir, 'models')
    
    for epoch in range(num_epochs):
        if logger:
            logger.info(f'Starting epoch {epoch + 1}/{num_epochs}')
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch + 1, print_freq, logger
        )
        
        # Validate
        val_metrics = None
        if val_loader:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch + 1, logger
            )
        
        # Update learning rate
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                if val_metrics:
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step(train_metrics['loss'])
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics tracker
        metrics_tracker.update(
            train_loss=train_metrics['loss'],
            val_loss=val_metrics['loss'] if val_metrics else None,
            lr=current_lr
        )
        
        # Check for best model
        is_best = False
        if val_metrics:
            val_acc = val_metrics['top1_accuracy']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                is_best = True
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
        else:
            # Use train accuracy if no validation set
            train_acc = train_metrics['top1_accuracy']
            if train_acc > best_val_acc:
                best_val_acc = train_acc
                is_best = True
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or is_best or epoch == num_epochs - 1:
            checkpoint_metrics = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'] if val_metrics else None,
                'val_acc': val_metrics['top1_accuracy'] if val_metrics else None,
                'learning_rate': current_lr
            }
            
            checkpoint_path = os.path.join(models_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, 
                checkpoint_metrics, checkpoint_path, is_best
            )
            
            if is_best and logger:
                logger.info(f'New best model saved with validation accuracy: {best_val_acc:.3f}')
        
        # Early stopping
        if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
            if logger:
                logger.info(f'Early stopping triggered after {early_stopping_patience} epochs without improvement')
            break
        
        if logger:
            logger.info(f'Epoch {epoch + 1} completed. Train Loss: {train_metrics["loss"]:.4f}' +
                       (f', Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["top1_accuracy"]:.2f}%' 
                        if val_metrics else ''))
    
    return metrics_tracker

def count_parameters(model):
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params