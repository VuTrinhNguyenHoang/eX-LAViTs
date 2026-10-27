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
from src.utils.training import load_checkpoint, load_config, count_parameters
from src.utils.metrics import compute_loss_and_accuracy, compute_classification_metrics
from src.utils.visualization import plot_confusion_matrix
from datasets.cifar10 import get_cifar10_dataloaders, get_cifar10_info

def load_eval_config(config_path):
    """Load evaluation configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def create_args_from_eval_config(config):
    """Create args namespace from evaluation config dictionary."""
    args = SimpleNamespace()
    
    # Model/checkpoint arguments
    model_config = config.get('model', {})
    args.checkpoint = model_config.get('checkpoint', None)
    args.config = model_config.get('config', None)
    
    # Data arguments
    data_config = config.get('data', {})
    args.data_dir = data_config.get('data_dir', './data')
    args.batch_size = data_config.get('batch_size', 128)
    args.num_workers = data_config.get('num_workers', 4)
    args.pin_memory = data_config.get('pin_memory', True)
    
    # Output arguments
    output_config = config.get('output', {})
    args.output_dir = output_config.get('output_dir', None)
    args.save_plots = output_config.get('save_plots', True)
    args.save_predictions = output_config.get('save_predictions', False)
    
    # Device arguments
    device_config = config.get('device', {})
    args.device = device_config.get('device', 'auto')
    
    return args

def create_model_from_config(config):
    """Create model from configuration."""
    model_kwargs = {
        'img_size': config.get('img_size', 32),
        'patch_size': config.get('patch_size', 4),
        'in_chans': 3,
        'num_classes': config.get('num_classes', 10),  # CIFAR-10
        'embed_dim': config.get('embed_dim', 128),
        'depth': config.get('depth', 6),
        'num_heads': config.get('num_heads', 8),
        'mlp_ratio': config.get('mlp_ratio', 4.0),
        'drop_rate': config.get('dropout', 0.1),
        'attn_drop_rate': config.get('attention_dropout', 0.0),
        'proj_drop_rate': config.get('projection_dropout', 0.0),
        'qkv_bias': config.get('qkv_bias', True),
        'act': config.get('activation', 'gelu'),
        'layer_norm_eps': config.get('layer_norm_eps', 1e-6)
    }
    
    model_type = config.get('model_type', 'StandardViT')
    if model_type.lower() == 'standardvit':
        model = StandardViT(**model_kwargs)
    elif model_type.lower() == 'lavit':
        model_kwargs['kernel'] = config.get('linear_attention_kernel', 'elu')
        model = LAViT(**model_kwargs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model

def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate model on test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    # Compute basic metrics
    results = compute_loss_and_accuracy(
        model, test_loader, criterion, device, return_predictions=True
    )
    
    # Compute detailed classification metrics
    classification_metrics = compute_classification_metrics(
        results['predictions'],
        results['targets'],
        class_names=class_names,
        return_confusion_matrix=True
    )
    
    # Combine results
    eval_results = {
        'loss': results['loss'],
        'top1_accuracy': results['top1_accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'macro_precision': classification_metrics['macro_precision'],
        'macro_recall': classification_metrics['macro_recall'],
        'macro_f1': classification_metrics['macro_f1'],
        'weighted_precision': classification_metrics['weighted_precision'],
        'weighted_recall': classification_metrics['weighted_recall'],
        'weighted_f1': classification_metrics['weighted_f1'],
        'predictions': results['predictions'],
        'targets': results['targets'],
        'confusion_matrix': classification_metrics['confusion_matrix'],
        'classification_report': classification_metrics['classification_report']
    }
    
    return eval_results

def print_evaluation_results(results, model_name="Model"):
    """Print evaluation results in a formatted way."""
    print(f"\n{model_name} Evaluation Results:")
    print("-" * 50)
    print(f"Loss: {results['loss']:.4f}")
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"Macro Precision: {results['macro_precision']:.2f}%")
    print(f"Macro Recall: {results['macro_recall']:.2f}%")
    print(f"Macro F1-Score: {results['macro_f1']:.2f}%")
    print(f"Weighted Precision: {results['weighted_precision']:.2f}%")
    print(f"Weighted Recall: {results['weighted_recall']:.2f}%")
    print(f"Weighted F1-Score: {results['weighted_f1']:.2f}%")
    print("-" * 50)

def save_evaluation_results(results, save_path, model_info=None):
    """Save evaluation results to JSON file."""
    # Remove numpy arrays and non-serializable items
    serializable_results = {
        'loss': results['loss'],
        'top1_accuracy': results['top1_accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'macro_precision': results['macro_precision'],
        'macro_recall': results['macro_recall'],
        'macro_f1': results['macro_f1'],
        'weighted_precision': results['weighted_precision'],
        'weighted_recall': results['weighted_recall'],
        'weighted_f1': results['weighted_f1']
    }
    
    if model_info:
        serializable_results['model_info'] = model_info
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained ViT models on CIFAR-10')
    parser.add_argument('--config', type=str, default='configs/eval_config.yaml',
                       help='Path to evaluation configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint file (overrides config)')
    
    cmd_args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(cmd_args.config):
        raise FileNotFoundError(f"Configuration file not found: {cmd_args.config}")
    
    print(f"Loading configuration from: {cmd_args.config}")
    config = load_eval_config(cmd_args.config)
    
    # Convert config to args namespace
    args = create_args_from_eval_config(config)
    
    # Override checkpoint if provided via command line
    if cmd_args.checkpoint:
        args.checkpoint = cmd_args.checkpoint
    
    # Validate that checkpoint is provided
    if not args.checkpoint:
        raise ValueError("Checkpoint path must be provided either in config file or via --checkpoint argument")
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Determine checkpoint directory
    checkpoint_dir = os.path.dirname(args.checkpoint)
    run_dir = os.path.dirname(checkpoint_dir) if os.path.basename(checkpoint_dir) == 'models' else checkpoint_dir
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        # Try to find config in the run directory
        possible_configs = [
            os.path.join(run_dir, 'configs', 'config.yaml'),
            os.path.join(run_dir, 'config.yaml'),
            os.path.join(checkpoint_dir, 'config.yaml')
        ]
        config_path = None
        for path in possible_configs:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            raise FileNotFoundError(f"Could not find config file. Searched: {possible_configs}")
    
    print(f"Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Create model
    print(f"Creating {config.get('model_type', 'Unknown')} model...")
    model = create_model_from_config(config)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Model created with {total_params:,} total parameters ({trainable_params:,} trainable)")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, model, device=device)
    
    model = model.to(device)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")
    
    # Create test data loader
    print("Creating test data loader...")
    _, _, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        img_size=config.get('img_size', 32),
        batch_size=args.batch_size,
        val_split=0.0,  # No validation split needed for evaluation
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        train_augmentation=False,  # No augmentation for test
        normalize=config.get('normalize', True),
        download=True
    )
    
    print(f"Test batches: {len(test_loader)}")
    
    # Get class names
    cifar10_info = get_cifar10_info()
    class_names = cifar10_info['classes']
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_model(model, test_loader, device, class_names)
    
    # Print results
    model_name = f"{config.get('model_type', 'Model')} (Epoch {checkpoint['epoch']})"
    print_evaluation_results(eval_results, model_name)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = run_dir
    
    # Save evaluation results
    model_info = {
        'model_type': config.get('model_type', 'Unknown'),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'checkpoint_epoch': checkpoint['epoch'],
        'img_size': config.get('img_size', 32),
        'patch_size': config.get('patch_size', 4),
        'embed_dim': config.get('embed_dim', 128),
        'num_heads': config.get('num_heads', 8),
        'depth': config.get('depth', 6)
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    save_evaluation_results(eval_results, results_path, model_info)
    print(f"Saved evaluation results to: {results_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_path = os.path.join(output_dir, 'predictions.json')
        predictions_data = {
            'predictions': eval_results['predictions'].tolist(),
            'targets': eval_results['targets'].tolist(),
            'class_names': class_names
        }
        with open(predictions_path, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"Saved predictions to: {predictions_path}")
    
    # Generate and save plots
    if args.save_plots:
        print("Generating plots...")
        plots_dir = os.path.join(output_dir, 'evaluation_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Confusion matrix
        fig = plot_confusion_matrix(
            eval_results['targets'],
            eval_results['predictions'],
            class_names=class_names,
            normalize=False
        )
        fig.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), 
                   bbox_inches='tight', dpi=300)
        
        # Normalized confusion matrix
        fig = plot_confusion_matrix(
            eval_results['targets'],
            eval_results['predictions'],
            class_names=class_names,
            normalize=True
        )
        fig.savefig(os.path.join(plots_dir, 'confusion_matrix_normalized.png'), 
                   bbox_inches='tight', dpi=300)
        
        print(f"Saved plots to: {plots_dir}")
    
    # Print per-class results
    print("\nPer-class Classification Report:")
    print("-" * 70)
    report = eval_results['classification_report']
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:12}: Precision: {metrics['precision']*100:5.1f}% | "
                  f"Recall: {metrics['recall']*100:5.1f}% | "
                  f"F1-Score: {metrics['f1-score']*100:5.1f}% | "
                  f"Support: {metrics['support']:4d}")
    
    print("\nEvaluation completed successfully!")

if __name__ == '__main__':
    main()