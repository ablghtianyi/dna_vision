
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import h5py
import io
import os
import random
from typing import Any, Callable, Optional, Tuple
from PIL import Image

import torch
from torchvision.datasets import VisionDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import decode_image
from torchvision.transforms import v2 as T  # Use v2 transforms
from typing import Dict, Tuple, List, Optional

from torch.utils.data import default_collate

# Cutmix augmentation
NUM_CLASSES=1000

cutmix_or_mixup = T.RandomChoice(
    [T.CutMix(num_classes=NUM_CLASSES),
     T.MixUp(alpha=0.8, num_classes=NUM_CLASSES)]
)

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))


class ImageNetDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str,
                 label_file: Optional[str] = None,
                 image_size: int = 224,
                 dataset_fraction: float = 1.0):
        """
        Args:
            root_dir (str): Path to the dataset root.
            split (str): 'train', 'val', or 'test'.
            label_file (str, optional): Path to labels.txt (not needed for test split).
            image_size (int): Size to resize images to (both height and width).
            dataset_fraction (float): Fraction of the dataset to use (0.0 to 1.0).
        """
        self.root_dir = os.path.join(root_dir, split)
        self.split = split
        self.image_size = image_size

        if split == 'test':
            self.image_paths = [
                os.path.join(self.root_dir, fname)
                for fname in os.listdir(self.root_dir)
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')) # Handle multiple extensions
            ]
            self.labels = None
        else:
            self.folder_to_label: Dict[str, Tuple[int, str]] = {}
            with open(label_file, 'r') as f:
                for idx, line in enumerate(f):
                    folder, text_label = line.strip().split(',')
                    folder = folder.strip()
                    text_label = text_label.strip()
                    self.folder_to_label[folder] = (idx, text_label)

            all_image_paths = []
            all_labels = []
            for folder in os.listdir(self.root_dir):
                if folder in self.folder_to_label:
                    folder_path = os.path.join(self.root_dir, folder)
                    for img_name in os.listdir(folder_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            all_image_paths.append(os.path.join(folder_path, img_name))
                            all_labels.append(self.folder_to_label[folder])

            if dataset_fraction < 1.0:
                num_samples = int(len(all_image_paths) * dataset_fraction)
                indices = random.sample(range(len(all_image_paths)), num_samples)
                self.image_paths = [all_image_paths[i] for i in indices]
                self.labels = [all_labels[i] for i in indices]
            else:
                self.image_paths = all_image_paths
                self.labels = all_labels

        if split == 'train':
            self.transform = T.Compose([
                T.RandomResizedCrop(224, scale=(0.05, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.AutoAugment(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            ])
        else:  # val and test
            self.transform = T.Compose([
                T.Resize(int(image_size * 256 / 224), antialias=True),
                T.CenterCrop(image_size),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int], Optional[str]]:
        img_path = self.image_paths[idx]
        image = decode_image(img_path, mode='RGB')  # Read directly into a tensor
        image = self.transform(image)  # Apply transforms

        if self.split == 'test':
            return image, None, None

        label_idx, _ = self.labels[idx]
        return image, label_idx


# Should only be used on aws with copying files to scratch of the GPU allocations
class H5VisionDataset(VisionDataset):
    def __init__(self, 
                 root_dir: str,
                 split: str,
                 image_size: int = 224):
        super().__init__(root_dir)
        self.root_dir = root_dir + split
        self.split = split
        self.image_size = image_size

        if 'train' in split:
            self.transform = T.Compose([
                T.PILToTensor(),
                T.RandomResizedCrop(224, scale=(0.05, 1.0), antialias=True),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
                T.AutoAugment(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            ])
        else:  # val and test
            self.transform = T.Compose([
                T.PILToTensor(),
                T.Resize(int(image_size * 256 / 224), antialias=True),
                T.CenterCrop(image_size),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        hf = h5py.File(self.root_dir, "r")
        self.image_bytes = hf["image_bytes"]
        self.targets = hf["targets"]
        self.n_samples = len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = Image.open(io.BytesIO(self.image_bytes[index])).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.n_samples
    

# DataLoader creation remains the same
def get_imagenet_dataloaders(
    root_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
    image_size: int = 224,
    dataset_fraction: float = 1.0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    h5py: bool = False,  # use only on aws
    mixup: bool = True,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Load label
    label_file = os.path.join(root_dir, 'labels.txt')

    if h5py is False:
        train_dataset = ImageNetDataset(root_dir, 'train', label_file, image_size=image_size, dataset_fraction=dataset_fraction)
        val_dataset = ImageNetDataset(root_dir, 'val', label_file, image_size=image_size, dataset_fraction=dataset_fraction)
        test_dataset = ImageNetDataset(root_dir, 'test', image_size=image_size)

        if distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size // world_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn if mixup else None,  # For cutmix
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=int(1.5 * batch_size // world_size),
            shuffle=False,
            sampler=val_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=int(1.5 * batch_size // world_size),
            shuffle=False,
            sampler=test_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        train_dataset = H5VisionDataset(root_dir, '/train.hdf5', image_size=image_size)
        val_dataset = H5VisionDataset(root_dir, '/val.hdf5', image_size=image_size)
        if distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size // world_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
            collate_fn=collate_fn if mixup else None,  # For cutmix
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=int(1.5 * batch_size // world_size),
            shuffle=False,
            sampler=val_sampler,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
        test_loader = None
    
    return train_loader, val_loader, test_loader


def get_imagenet_val_loader(
    root_dir: str,
    batch_size: int = 128,
    num_workers: int = 8,
    image_size: int = 224,
    dataset_fraction: float = 1.0,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    h5py: bool = False,  # use only on aws
    shuffle: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Load label
    label_file = os.path.join(root_dir, 'labels.txt')

    if h5py is False:
        val_dataset = ImageNetDataset(root_dir, 'val', label_file, image_size=image_size, dataset_fraction=dataset_fraction)

        if distributed:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            val_sampler = None

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size // world_size,
            shuffle=shuffle,
            sampler=val_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    else:
        val_dataset = H5VisionDataset(root_dir, '/val.hdf5', image_size=image_size)
        if distributed:
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size // world_size,
            shuffle=shuffle,
            sampler=val_sampler,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    
    return val_loader