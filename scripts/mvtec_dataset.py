"""
MVTec AD Dataset Loader for PyTorch.

This module provides a PyTorch Dataset class for loading MVTec AD data
for anomaly detection tasks.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
import torch
from torch.utils.data import Dataset
from PIL import Image


class MVTecADDataset(Dataset):
    """
    MVTec AD Dataset for anomaly detection.

    The dataset structure should be:
    root/
        category/
            train/
                good/
                    *.png
            test/
                good/
                    *.png
                defect_type_1/
                    *.png
                ...
            ground_truth/
                defect_type_1/
                    *.png
                ...

    Args:
        root: Root directory of MVTec AD dataset
        category: Category name (e.g., 'bottle', 'cable', etc.)
        split: 'train' or 'test'
        transform: Optional transform to be applied on images
        mask_transform: Optional transform to be applied on masks
    """

    CATEGORIES = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    def __init__(
        self,
        root: str,
        category: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        assert category in self.CATEGORIES, f"category must be one of {self.CATEGORIES}"

        self.root = Path(root)
        self.category = category
        self.split = split
        self.transform = transform
        self.mask_transform = mask_transform

        self.category_path = self.root / category
        self.split_path = self.category_path / split
        self.gt_path = self.category_path / "ground_truth"

        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.mask_paths = []
        self.defect_types = []

        self._load_dataset()

    def _load_dataset(self):
        """Load all image paths and corresponding labels."""
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split path not found: {self.split_path}")

        # Get all defect type folders
        defect_folders = sorted([d for d in self.split_path.iterdir() if d.is_dir()])

        for defect_folder in defect_folders:
            defect_type = defect_folder.name
            is_good = (defect_type == 'good')

            # Get all images in this defect type
            image_files = sorted(list(defect_folder.glob('*.png')))

            for img_path in image_files:
                self.image_paths.append(img_path)
                self.labels.append(0 if is_good else 1)  # 0 = normal, 1 = anomaly
                self.defect_types.append(defect_type)

                # Find corresponding mask for test anomalies
                if not is_good and self.split == 'test':
                    mask_path = self.gt_path / defect_type / (img_path.stem + '_mask.png')
                    self.mask_paths.append(mask_path if mask_path.exists() else None)
                else:
                    self.mask_paths.append(None)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        """
        Get item by index.

        Returns:
            dict containing:
                - image: PIL Image or transformed tensor
                - label: 0 for normal, 1 for anomaly
                - mask: ground truth mask (if available, else None)
                - defect_type: defect type name
                - image_path: path to the image file
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        defect_type = self.defect_types[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Load mask if available
        mask = None
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert('L')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if mask is not None and self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {
            'image': image,
            'label': label,
            'mask': mask,
            'defect_type': defect_type,
            'image_path': str(img_path),
        }

    def get_normal_samples(self) -> List[int]:
        """Get indices of all normal (good) samples."""
        return [i for i, label in enumerate(self.labels) if label == 0]

    def get_anomaly_samples(self) -> List[int]:
        """Get indices of all anomaly samples."""
        return [i for i, label in enumerate(self.labels) if label == 1]

    def get_defect_types(self) -> List[str]:
        """Get list of unique defect types in dataset."""
        return sorted(list(set(self.defect_types)))


def get_mvtec_transforms(image_size: int = 224):
    """
    Get standard transforms for MVTec AD dataset compatible with DINOv3.

    Args:
        image_size: Target image size

    Returns:
        Transform for images
    """
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return transform


def get_mask_transform(image_size: int = 224):
    """
    Get transform for ground truth masks.

    Args:
        image_size: Target mask size

    Returns:
        Transform for masks
    """
    from torchvision.transforms import v2

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
    ])

    return transform


if __name__ == "__main__":
    # Example usage
    dataset = MVTecADDataset(
        root="./mvtec_ad",
        category="carpet",
        split="train",
        transform=get_mvtec_transforms(224),
    )

    print(f"Dataset: {dataset.category} - {dataset.split}")
    print(f"Total samples: {len(dataset)}")
    print(f"Normal samples: {len(dataset.get_normal_samples())}")
    print(f"Anomaly samples: {len(dataset.get_anomaly_samples())}")
    print(f"Defect types: {dataset.get_defect_types()}")

    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Label: {sample['label']}")
    print(f"  Defect type: {sample['defect_type']}")
