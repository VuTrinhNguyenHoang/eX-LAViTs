import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

def get_cifar10_dataloaders(
    data_dir: str = './data',
    img_size: int = 32,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create CIFAR-10 train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory to save/load CIFAR-10 data
        img_size: Target image size (will resize if different from 32)
        batch_size: Batch size for dataloaders
        val_split: Fraction of training data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        download: Whether to download CIFAR-10 if not found
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # CIFAR-10 statistics
    cifar10_mean = (0.485, 0.456, 0.406)
    cifar10_std = (0.229, 0.224, 0.225)
    
    # Train transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    # Test/Validate transforms
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=test_transform
    )
    
    # Split training data into train and validation
    if val_split > 0:
        num_train = len(train_dataset)
        num_val = int(num_train * val_split)
        num_train = num_train - num_val
        
        train_subset, val_subset = random_split(
            train_dataset, 
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create validation dataset with test transforms (no augmentation)
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=False,
            transform=test_transform
        )
        
        # Apply the validation split indices to the non-augmented dataset
        val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)
        train_dataset = train_subset
    else:
        val_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
    else:
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader


def get_cifar10_classes():
    """Get CIFAR-10 class names."""
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def get_cifar10_info():
    """Get basic information about CIFAR-10 dataset."""
    return {
        'num_classes': 10,
        'classes': get_cifar10_classes(),
        'input_shape': (3, 32, 32),
        'train_size': 50000,
        'test_size': 10000,
        'mean': (0.4914, 0.4822, 0.4465),
        'std': (0.2023, 0.1994, 0.2010)
    }