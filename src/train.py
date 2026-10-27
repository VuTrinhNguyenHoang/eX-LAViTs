import argparse
import os
import sys
import json
import yaml
import torch
import torch.nn as nn
from types import SimpleNamespace

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vit import StandardViT, LAViT
from src.utils.training import (
    create_run_directory, setup_logging, save_config, get_optimizer, 
    get_scheduler, train_model, count_parameters, get_model_info, load_checkpoint
)
from src.utils.metrics import compute_loss_and_accuracy, compute_classification_metrics
from src.utils.visualization import save_all_plots, create_training_summary_plot
from datasets import get_dataset, DATASETS

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def dict_to_namespace(d):
    """Convert nested dictionary to nested namespace for dot notation access."""
    namespace = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(namespace, key, dict_to_namespace(value))
        else:
            setattr(namespace, key, value)
    return namespace

def create_args_from_config(config):
    """Create args namespace from config dictionary with flattened structure."""
    args = SimpleNamespace()
    
    # Dataset arguments
    dataset_config = config.get('dataset', {})
    args.dataset_name = dataset_config.get('name', 'cifar10')
    args.data_dir = dataset_config.get('data_dir', './data')
    args.val_split = dataset_config.get('val_split', 0.1)
    args.num_workers = dataset_config.get('num_workers', 4)
    args.pin_memory = dataset_config.get('pin_memory', True)
    
    # Model arguments
    model_config = config.get('model', {})
    args.model_type = model_config.get('model_type', 'StandardViT')
    args.img_size = model_config.get('img_size', 32)
    args.num_classes = model_config.get('num_classes', 10)
    args.patch_size = model_config.get('patch_size', 4)
    args.embed_dim = model_config.get('embed_dim', 128)
    args.depth = model_config.get('depth', 6)
    args.num_heads = model_config.get('num_heads', 8)
    args.mlp_ratio = model_config.get('mlp_ratio', 4.0)
    args.dropout = model_config.get('dropout', 0.1)
    args.attention_dropout = model_config.get('attention_dropout', 0.0)
    args.projection_dropout = model_config.get('projection_dropout', 0.0)
    args.qkv_bias = model_config.get('qkv_bias', True)
    args.activation = model_config.get('activation', 'gelu')
    args.layer_norm_eps = model_config.get('layer_norm_eps', 1e-6)
    args.linear_attention_kernel = model_config.get('linear_attention_kernel', 'elu')
    
    # Training arguments
    training_config = config.get('training', {})
    args.epochs = training_config.get('epochs', 100)
    args.batch_size = training_config.get('batch_size', 128)
    args.learning_rate = training_config.get('learning_rate', 1e-3)
    args.weight_decay = training_config.get('weight_decay', 1e-4)
    args.optimizer = training_config.get('optimizer', 'adamw')
    args.scheduler = training_config.get('scheduler', 'cosine')
    args.warmup_epochs = training_config.get('warmup_epochs', 10)
    args.min_lr = training_config.get('min_lr', 1e-6)
    args.save_freq = training_config.get('save_freq', 10)
    args.print_freq = training_config.get('print_freq', 50)
    args.early_stopping = training_config.get('early_stopping', None)
    
    # Experiment arguments
    experiment_config = config.get('experiment', {})
    args.experiment_dir = experiment_config.get('experiment_dir', 'experiments')
    args.run_name = experiment_config.get('run_name', None)
    args.resume = experiment_config.get('resume', None)
    args.seed = experiment_config.get('seed', 42)
    args.device = experiment_config.get('device', 'auto')
    
    return args

def create_model(args):
    """Create model based on arguments."""
    model_kwargs = {
        'img_size': args.img_size,
        'patch_size': args.patch_size,
        'in_chans': 3,
        'num_classes': args.num_classes,
        'embed_dim': args.embed_dim,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'drop_rate': args.dropout,
        'attn_drop_rate': args.attention_dropout,
        'proj_drop_rate': args.projection_dropout,
        'qkv_bias': args.qkv_bias,
        'act': args.activation,
        'layer_norm_eps': args.layer_norm_eps
    }
    
    if args.model_type.lower() == 'standardvit':
        model = StandardViT(**model_kwargs)
    elif args.model_type.lower() == 'lavit':
        model_kwargs['kernel'] = args.linear_attention_kernel
        model = LAViT(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train ViT models on various datasets (CIFAR-10/100, Food-101, STL-10)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(cmd_args.config):
        raise FileNotFoundError(f"Configuration file not found: {cmd_args.config}")
    
    print(f"Loading configuration from: {cmd_args.config}")
    config = load_config(cmd_args.config)
    
    # Convert config to args namespace
    args = create_args_from_config(config)
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create run directory
    run_dir = create_run_directory(args.experiment_dir, args.run_name)
    print(f"Created run directory: {run_dir}")
    
    # Setup logging
    logger = setup_logging(os.path.join(run_dir, 'logs'))
    logger.info(f"Starting training with arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    # Save configuration
    config_path = os.path.join(run_dir, 'configs', 'config.yaml')
    save_config(config, config_path)
    logger.info(f"Saved configuration to: {config_path}")
    
    # Create data loaders
    logger.info(f"Creating data loaders for {args.dataset_name}...")
    
    try:
        loaders, dataset_info = get_dataset(
            args.dataset_name,
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            download=True
        )
        
        train_loader, val_loader, test_loader = loaders
        
        # Validate that num_classes matches dataset
        if args.num_classes != dataset_info['num_classes']:
            logger.warning(f"Model num_classes ({args.num_classes}) doesn't match dataset ({dataset_info['num_classes']}). Using dataset value.")
            args.num_classes = dataset_info['num_classes']
        
        # Validate image size
        expected_shape = dataset_info['input_shape']
        if expected_shape[1] != args.img_size or expected_shape[2] != args.img_size:
            logger.warning(f"Model img_size ({args.img_size}) doesn't match dataset default ({expected_shape[1]}x{expected_shape[2]})")
            
    except Exception as e:
        logger.error(f"Failed to load dataset {args.dataset_name}: {e}")
        raise
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader) if val_loader else 0}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info(f"Creating {args.model_type} model...")
    model = create_model(args)
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create optimizer
    optimizer = get_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler_kwargs = {}
    if args.scheduler == 'cosine':
        scheduler_kwargs = {
            'T_max': args.epochs,
            'eta_min': args.min_lr
        }
    
    scheduler = get_scheduler(optimizer, args.scheduler, **scheduler_kwargs)
    
    logger.info(f"Created optimizer: {type(optimizer).__name__}")
    logger.info(f"Created scheduler: {type(scheduler).__name__ if scheduler else 'None'}")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed from epoch {start_epoch}")
    
    # Train model
    logger.info("Starting training...")
    metrics_tracker = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        run_dir=run_dir,
        save_freq=args.save_freq,
        print_freq=args.print_freq,
        early_stopping_patience=args.early_stopping,
        logger=logger
    )
    
    logger.info("Training completed!")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    
    test_results = compute_loss_and_accuracy(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    # Compute detailed classification metrics
    class_names = dataset_info.get('classes', [f'Class_{i}' for i in range(dataset_info['num_classes'])])
    
    classification_metrics = compute_classification_metrics(
        test_results['predictions'],
        test_results['targets'],
        class_names=class_names,
        return_confusion_matrix=True
    )
    
    # Log test results
    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_results['loss']:.4f}")
    logger.info(f"  Top-1 Accuracy: {test_results['top1_accuracy']:.2f}%")
    logger.info(f"  Macro F1-Score: {classification_metrics['macro_f1']:.2f}%")
    logger.info(f"  Weighted F1-Score: {classification_metrics['weighted_f1']:.2f}%")
    
    # Save test results
    results_path = os.path.join(run_dir, 'test_results.json')
    test_summary = {
        'test_loss': test_results['loss'],
        'test_top1_accuracy': test_results['top1_accuracy'],
        'test_macro_f1': classification_metrics['macro_f1'],
        'test_weighted_f1': classification_metrics['weighted_f1'],
        'model_params': total_params,
        'trainable_params': trainable_params
    }
    
    with open(results_path, 'w') as f:
        json.dump(test_summary, f, indent=2)
    
    logger.info(f"Saved test results to: {results_path}")
    
    # Generate and save plots
    logger.info("Generating plots...")
    save_all_plots(
        metrics_tracker=metrics_tracker,
        run_dir=run_dir,
        y_true=test_results['targets'],
        y_pred=test_results['predictions'],
        class_names=class_names
    )
    
    # Create comprehensive training summary plot with enhanced information
    model_info = get_model_info(model, args)
    
    # Add dataset information
    train_dataset_size = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'N/A'
    val_dataset_size = len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else 'N/A'
    
    model_info.update({
        'dataset_name': args.dataset_name,
        'train_samples': train_dataset_size,
        'val_samples': val_dataset_size,
        'dataset_classes': dataset_info['num_classes'],
        'dataset_shape': dataset_info['input_shape'],
    })
    
    summary_plot = create_training_summary_plot(metrics_tracker, model_info)
    summary_plot.savefig(
        os.path.join(run_dir, 'plots', 'training_summary.png'),
        bbox_inches='tight', dpi=300
    )
    
    logger.info(f"Saved training summary to: {os.path.join(run_dir, 'plots', 'training_summary.png')}")
    
    logger.info(f"Saved plots to: {os.path.join(run_dir, 'plots')}")
    
    # Print summary
    summary = metrics_tracker.get_summary()
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset_name.upper()}")
    print(f"Model: {args.model_type}")
    print(f"Run Directory: {run_dir}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Dataset Classes: {dataset_info['num_classes']}")
    print(f"Image Size: {args.img_size}x{args.img_size}")
    print(f"Best Validation Accuracy: {summary.get('best_val_accuracy', 0):.2f}% (Epoch {summary.get('best_val_accuracy_epoch', 0) + 1})")
    print(f"Final Test Accuracy: {test_results['top1_accuracy']:.2f}%")
    print(f"Test F1-Score (Macro): {classification_metrics['macro_f1']:.2f}%")
    
    # Add timing and memory information
    if summary.get('total_training_time'):
        print(f"Total Training Time: {summary.get('total_training_time', 0):.1f} seconds")
        print(f"Average Epoch Time: {summary.get('avg_epoch_time', 0):.2f} seconds")
    
    if summary.get('peak_memory_usage'):
        print(f"Peak Memory Usage: {summary.get('peak_memory_usage', 0):.0f} MB")
        print(f"Average Memory Usage: {summary.get('avg_memory_usage', 0):.0f} MB")
    
    print("="*60)
    
    logger.info("Training script completed successfully!")

if __name__ == '__main__':
    main()