import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

def get_cifar100_dataloaders(
    data_dir: str = './data',
    img_size: int = 32,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create CIFAR-100 train, validation, and test dataloaders."""
    
    # CIFAR-100 statistics
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    
    # Train transforms with stronger augmentation for more classes
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=download, transform=test_transform
    )
    
    # Split for validation
    if val_split > 0:
        num_train = len(train_dataset)
        num_val = int(num_train * val_split)
        num_train = num_train - num_val
        
        train_subset, val_subset = random_split(
            train_dataset, [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=False, transform=test_transform
        )
        val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)
        train_dataset = train_subset
    else:
        val_dataset = None
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=pin_memory, drop_last=False) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    
    return train_loader, val_loader, test_loader

def get_cifar100_info():
    """Get basic information about CIFAR-100 dataset."""
    return {
        'num_classes': 100,
        'input_shape': (3, 32, 32),
        'train_size': 50000,
        'test_size': 10000,
        'mean': (0.5071, 0.4867, 0.4408),
        'std': (0.2675, 0.2565, 0.2761)
    }