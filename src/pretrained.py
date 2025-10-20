import argparse
import os
import sys
import json
import yaml
import torch
import torch.nn as nn
from types import SimpleNamespace
import timm
import warnings

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.training import (
    create_run_directory, setup_logging, save_config, get_optimizer, 
    get_scheduler, train_model, count_parameters, get_model_info, load_checkpoint
)
from src.utils.metrics import compute_loss_and_accuracy, compute_classification_metrics
from src.utils.visualization import save_all_plots, create_training_summary_plot
from src.models.linear_vit import create_linear_attention_vit
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
    args.model_name = model_config.get('model_name', 'vit_small_patch16_224')
    args.pretrained = model_config.get('pretrained', True)
    args.use_linear_attention = model_config.get('use_linear_attention', False)
    args.linear_attention_kernel = model_config.get('linear_attention_kernel', 'elu')
    args.img_size = model_config.get('img_size', 32)
    args.num_classes = model_config.get('num_classes', 10)
    args.patch_size = model_config.get('patch_size', 4)
    args.embed_dim = model_config.get('embed_dim', 192)
    args.depth = model_config.get('depth', 8)
    args.num_heads = model_config.get('num_heads', 12)
    args.dropout = model_config.get('dropout', 0.1)
    args.drop_path_rate = model_config.get('drop_path_rate', 0.1)
    
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
    
    # Warmup
    args.warmup_last4_epochs = training_config.get('warmup_last4_epochs', 5)  # số epoch chỉ train 4 block cuối
    args.warmup_lr_mult = training_config.get('warmup_lr_mult', 0.5)

    # Experiment arguments
    experiment_config = config.get('experiment', {})
    args.experiment_dir = experiment_config.get('experiment_dir', 'experiments')
    args.run_name = experiment_config.get('run_name', None)
    args.resume = experiment_config.get('resume', None)
    args.seed = experiment_config.get('seed', 42)
    args.device = experiment_config.get('device', 'auto')
    
    return args

def create_pretrained_model(args, logger=None):
    """Create và configure một mô hình ViT pretrained."""
    
    if logger:
        logger.info(f"Creating pretrained model: {args.model_name}")
        logger.info(f"Pretrained weights: {args.pretrained}")
        logger.info(f"Use linear attention: {getattr(args, 'use_linear_attention', False)}")
        if getattr(args, 'use_linear_attention', False):
            logger.info(f"Linear attention kernel: {getattr(args, 'linear_attention_kernel', 'elu')}")
        logger.info(f"Target image size: {args.img_size}")
        logger.info(f"Target number of classes: {args.num_classes}")
    
    try:
        if getattr(args, 'use_linear_attention', False):
            # Sử dụng LinearMultiheadAttention
            model = create_linear_attention_vit(
                model_name=args.model_name,
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                img_size=args.img_size,
                drop_rate=args.dropout,
                drop_path_rate=getattr(args, 'drop_path_rate', 0.1),
                kernel=getattr(args, 'linear_attention_kernel', 'elu'),
                verbose=logger is not None
            )
        else:
            # Tạo mô hình standard với timm
            model = timm.create_model(
                args.model_name,
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                img_size=args.img_size,
                drop_rate=args.dropout,
                drop_path_rate=getattr(args, 'drop_path_rate', 0.1)
            )
        
        if logger:
            total_params, trainable_params = count_parameters(model)
            logger.info(f"Model created successfully!")
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create model {args.model_name}: {e}")
            logger.info("Available ViT models in timm:")
            available_models = timm.list_models('vit*', pretrained=True)
            for model_name in available_models[:10]:  # Show first 10
                logger.info(f"  - {model_name}")
        raise

def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def freeze_all_except_last_k_blocks(model: nn.Module, k: int = 4, logger=None):
    if not hasattr(model, 'blocks'):
        raise AttributeError("Model không có attribute 'blocks'.")
    n = len(model.blocks)
    for i, blk in enumerate(model.blocks):
        set_requires_grad(blk, i >= n - k)
    # giữ head và norm trainable
    for name in ['head', 'fc', 'norm']:
        if hasattr(model, name):
            set_requires_grad(getattr(model, name), True)
    # patch embed và pos_embed frozen
    if hasattr(model, 'patch_embed'):
        set_requires_grad(model.patch_embed, False)
    if hasattr(model, 'pos_embed') and isinstance(model.pos_embed, torch.nn.Parameter):
        model.pos_embed.requires_grad = False
    if logger:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[Warmup] Trainable params with last {k} blocks: {trainable_params:,}")

def unfreeze_all(model: nn.Module, logger=None):
    for p in model.parameters():
        p.requires_grad = True
    if logger:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"[Finetune] Trainable params (all): {trainable_params:,}")

def rebuild_optimizer_and_scheduler(model, args, epochs_for_phase, base_lr, logger=None):
    optimizer = get_optimizer(
        model,
        optimizer_name=args.optimizer,
        learning_rate=base_lr,
        weight_decay=args.weight_decay
    )
    scheduler_kwargs = {}
    if args.scheduler == 'cosine':
        scheduler_kwargs = {'T_max': epochs_for_phase, 'eta_min': args.min_lr}
    scheduler = get_scheduler(optimizer, args.scheduler, **scheduler_kwargs)
    if logger:
        logger.info(f"Built optimizer={type(optimizer).__name__}, lr={base_lr}, scheduler={type(scheduler).__name__ if scheduler else 'None'}, T_max={scheduler_kwargs.get('T_max', None)}")
    return optimizer, scheduler

def main():
    parser = argparse.ArgumentParser(description='Train pretrained ViT models on various datasets')
    parser.add_argument('--config', type=str, default='configs/cifar10_config.yaml',
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
    run_name = args.run_name if args.run_name else f"pretrained_{args.model_name}"
    run_dir = create_run_directory(args.experiment_dir, run_name)
    print(f"Created run directory: {run_dir}")
    
    # Setup logging
    logger = setup_logging(os.path.join(run_dir, 'logs'))
    logger.info(f"Starting training with pretrained ViT")
    logger.info(f"Configuration: {vars(args)}")
    logger.info(f"Using device: {device}")
    
    # Save configuration
    config_path = os.path.join(run_dir, 'configs', 'config.yaml')
    
    # Add pretrained model info to config
    config['model']['model_name'] = args.model_name
    config['model']['pretrained'] = args.pretrained
    
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
    
    # Create pretrained model
    logger.info(f"Creating pretrained ViT model...")
    model = create_pretrained_model(args, logger)
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model loaded with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # ===== Phase 1: Warm-up last 4 blocks =====
    warmup_epochs = int(max(0, args.warmup_last4_epochs))
    remaining_epochs = int(max(0, args.epochs - warmup_epochs))

    # Train model
    logger.info("Starting training...")
    if warmup_epochs > 0:
        logger.info(f"Phase 1: Warm-up last 4 blocks for {warmup_epochs} epochs")
        freeze_all_except_last_k_blocks(model, k=4, logger=logger)

        base_lr_phase1 = args.learning_rate * float(args.warmup_lr_mult)
        optimizer1, scheduler1 = rebuild_optimizer_and_scheduler(
            model, args, warmup_epochs, base_lr_phase1, logger
        )

        metrics_phase1 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer1,
            scheduler=scheduler1,
            device=device,
            num_epochs=warmup_epochs,
            run_dir=run_dir,
            save_freq=args.save_freq,
            print_freq=args.print_freq,
            early_stopping_patience=args.early_stopping,
            logger=logger
        )
        logger.info("Phase 1 completed.")

    if remaining_epochs > 0:
        logger.info(f"Phase 2: Unfreeze all and finetune for {remaining_epochs} epochs")
        unfreeze_all(model, logger=logger)
        base_lr_phase2 = args.learning_rate
        optimizer2, scheduler2 = rebuild_optimizer_and_scheduler(
            model, args, remaining_epochs, base_lr_phase2, logger
        )

        metrics_phase2 = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer2,
            scheduler=scheduler2,
            device=device,
            num_epochs=remaining_epochs,
            run_dir=run_dir,
            save_freq=args.save_freq,
            print_freq=args.print_freq,
            early_stopping_patience=args.early_stopping,
            logger=logger
        )
        logger.info("Phase 2 completed.")

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
        'model_name': args.model_name,
        'pretrained': args.pretrained,
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
        metrics_tracker=metrics_phase2,
        run_dir=run_dir,
        y_true=test_results['targets'],
        y_pred=test_results['predictions'],
        class_names=class_names
    )
    
    # Create comprehensive training summary plot
    model_info = {
        'model_name': args.model_name,
        'pretrained': args.pretrained,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'dataset_name': args.dataset_name,
        'train_samples': len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 'N/A',
        'val_samples': len(val_loader.dataset) if val_loader and hasattr(val_loader, 'dataset') else 'N/A',
        'dataset_classes': dataset_info['num_classes'],
        'dataset_shape': dataset_info['input_shape'],
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'scheduler': args.scheduler,
    }
    
    summary_plot = create_training_summary_plot(metrics_phase2, model_info)
    summary_plot.savefig(
        os.path.join(run_dir, 'plots', 'training_summary.png'),
        bbox_inches='tight', dpi=300
    )
    
    logger.info(f"Saved training summary to: {os.path.join(run_dir, 'plots', 'training_summary.png')}")
    logger.info(f"Saved plots to: {os.path.join(run_dir, 'plots')}")
    
    # Print summary
    summary = metrics_phase2.get_summary()
    print("\n" + "="*70)
    print("PRETRAINED ViT TRAINING SUMMARY")
    print("="*70)
    print(f"Dataset: {args.dataset_name.upper()}")
    print(f"Model: {args.model_name}")
    print(f"Pretrained: {'Yes' if args.pretrained else 'No'}")
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
    
    print("="*70)
    
    logger.info("Pretrained ViT training script completed successfully!")

if __name__ == '__main__':
    main()