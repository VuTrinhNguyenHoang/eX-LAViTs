import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

def get_food101_dataloaders(
    data_dir: str = './data',
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create Food-101 dataloaders.
    Food-101: 101 food classes, ~1000 images per class
    """
    
    # ImageNet statistics (commonly used for Food-101)
    food101_mean = (0.485, 0.456, 0.406)
    food101_std = (0.229, 0.224, 0.225)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(food101_mean, food101_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(food101_mean, food101_std)
    ])
    
    train_dataset = torchvision.datasets.Food101(
        root=data_dir, split='train', download=download, transform=train_transform
    )
    test_dataset = torchvision.datasets.Food101(
        root=data_dir, split='test', download=download, transform=test_transform
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
        val_dataset = torchvision.datasets.Food101(
            root=data_dir, split='train', download=False, transform=test_transform
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

def get_food101_info():
    return {
        'num_classes': 101,
        'input_shape': (3, 224, 224),
        'train_size': 75750,
        'test_size': 25250,
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    }