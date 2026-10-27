from .cifar10 import get_cifar10_dataloaders, get_cifar10_info
from .cifar100 import get_cifar100_dataloaders, get_cifar100_info
from .food101 import get_food101_dataloaders, get_food101_info
from .stl10 import get_stl10_dataloaders, get_stl10_info

__all__ = [
    'get_cifar10_dataloaders', 'get_cifar10_info',
    'get_cifar100_dataloaders', 'get_cifar100_info', 
    'get_food101_dataloaders', 'get_food101_info',
    'get_stl10_dataloaders', 'get_stl10_info',
    'get_dataset', 'DATASETS'
]

# Dataset registry for easy access
DATASETS = {
    'cifar10': {
        'loader_fn': get_cifar10_dataloaders,
        'info_fn': get_cifar10_info,
        'default_img_size': 32,
    },
    'cifar100': {
        'loader_fn': get_cifar100_dataloaders,
        'info_fn': get_cifar100_info,
        'default_img_size': 32,
    },
    'food101': {
        'loader_fn': get_food101_dataloaders,
        'info_fn': get_food101_info,
        'default_img_size': 224,
    },
    'stl10': {
        'loader_fn': get_stl10_dataloaders,
        'info_fn': get_stl10_info,
        'default_img_size': 96,
    },
}

def get_dataset(dataset_name: str, **kwargs):
    """
    Factory function to get dataset loaders and info.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'food101', 'stl10')
        **kwargs: Additional arguments to pass to the dataset loader
        
    Returns:
        tuple: (train_loader, val_loader, test_loader), dataset_info
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASETS:
        available = list(DATASETS.keys())
        raise ValueError(f"Dataset '{dataset_name}' not supported. Available datasets: {available}")
    
    config = DATASETS[dataset_name]
    
    # Set default image size if not provided
    if 'img_size' not in kwargs:
        kwargs['img_size'] = config['default_img_size']
    
    try:
        # Load dataset
        loaders = config['loader_fn'](**kwargs)
        info = config['info_fn']()
        
        return loaders, info
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset '{dataset_name}': {str(e)}")