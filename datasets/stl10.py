import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple

def get_stl10_dataloaders(
    data_dir: str = './data',
    img_size: int = 96,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create STL-10 dataloaders.
    STL-10: 10 classes, 96x96 images, 500 train + 800 test + 100k unlabeled
    """
    
    # STL-10 statistics
    stl10_mean = (0.4467, 0.4398, 0.4066)
    stl10_std = (0.2603, 0.2566, 0.2713)
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(stl10_mean, stl10_std)
    ])
    
    train_dataset = torchvision.datasets.STL10(
        root=data_dir, split='train', download=download, transform=train_transform
    )
    test_dataset = torchvision.datasets.STL10(
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
        val_dataset = torchvision.datasets.STL10(
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

def get_stl10_info():
    return {
        'num_classes': 10,
        'input_shape': (3, 96, 96),
        'train_size': 5000,
        'test_size': 8000,
        'unlabeled_size': 100000,
        'mean': (0.4467, 0.4398, 0.4066),
        'std': (0.2603, 0.2566, 0.2713)
    }