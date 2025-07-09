from typing import Tuple

import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2 as T  # Use v2 transforms

from torch.utils.data import default_collate

# Cutmix augmentation
NUM_CLASSES=10

cutmix_or_mixup = T.RandomChoice(
    [T.CutMix(num_classes=NUM_CLASSES),
     T.MixUp(alpha=0.8, num_classes=NUM_CLASSES)]
)

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


# DataLoader creation remains the same
def get_cifar10_dataloaders(
    root_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_transform = T.Compose([
                T.PILToTensor(),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomCrop(32, padding=4),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            ])
    val_transform = T.Compose([
                T.PILToTensor(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
            ])
    train_dataset = CIFAR10(root_dir, train=True, download=False, transform=train_transform)
    val_dataset = CIFAR10(root_dir, train=False, download=False, transform=val_transform)

    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size // world_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_fn,  # For cutmix
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(1.5 * batch_size // world_size),
        shuffle=False,
        sampler=val_sampler,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    
    return train_loader, val_loader, None  # For compatability