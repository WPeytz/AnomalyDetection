"""
Script to download and organize the MVTec AD dataset.

MVTec AD is an anomaly detection dataset with 15 categories of objects and textures.
Each category contains normal training images and test images with various defect types.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for download."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar."""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_mvtec_ad(root_dir="./mvtec_ad"):
    """
    Download and extract MVTec AD dataset.

    Args:
        root_dir: Root directory where the dataset will be stored
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # MVTec AD categories
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]

    base_url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/"

    # Download URLs for each category
    download_urls = {
        'bottle': f'{base_url}420937370-1629951468/bottle.tar.xz',
        'cable': f'{base_url}420937413-1629951498/cable.tar.xz',
        'capsule': f'{base_url}420937454-1629951595/capsule.tar.xz',
        'carpet': f'{base_url}420937484-1629951672/carpet.tar.xz',
        'grid': f'{base_url}420937487-1629951814/grid.tar.xz',
        'hazelnut': f'{base_url}420937545-1629951845/hazelnut.tar.xz',
        'leather': f'{base_url}420937607-1629951964/leather.tar.xz',
        'metal_nut': f'{base_url}420937637-1629952063/metal_nut.tar.xz',
        'pill': f'{base_url}420938129-1629953099/pill.tar.xz',
        'screw': f'{base_url}420938130-1629953152/screw.tar.xz',
        'tile': f'{base_url}420938133-1629953189/tile.tar.xz',
        'toothbrush': f'{base_url}420938134-1629953256/toothbrush.tar.xz',
        'transistor': f'{base_url}420938166-1629953277/transistor.tar.xz',
        'wood': f'{base_url}420938383-1629953354/wood.tar.xz',
        'zipper': f'{base_url}420938385-1629953449/zipper.tar.xz',
    }

    print(f"Downloading MVTec AD dataset to {root_dir}")
    print(f"Total categories: {len(categories)}\n")

    for category in categories:
        category_path = root_dir / category
        tar_path = root_dir / f"{category}.tar.xz"

        # Check if category already exists
        if category_path.exists() and len(list(category_path.iterdir())) > 0:
            print(f"✓ {category}: Already downloaded and extracted")
            continue

        # Download
        if not tar_path.exists():
            print(f"⬇ Downloading {category}...")
            try:
                download_url(download_urls[category], str(tar_path))
            except Exception as e:
                print(f"✗ Failed to download {category}: {e}")
                continue

        # Extract
        print(f"📦 Extracting {category}...")
        try:
            with tarfile.open(tar_path, 'r:xz') as tar:
                tar.extractall(root_dir)
            print(f"✓ {category}: Completed\n")

            # Remove tar file to save space
            tar_path.unlink()
        except Exception as e:
            print(f"✗ Failed to extract {category}: {e}\n")

    print("\n✓ MVTec AD dataset setup complete!")
    print(f"Dataset location: {root_dir.absolute()}")

    # Print dataset structure
    print("\nDataset structure:")
    for category in categories:
        category_path = root_dir / category
        if category_path.exists():
            train_path = category_path / "train"
            test_path = category_path / "test"
            ground_truth_path = category_path / "ground_truth"

            train_count = len(list(train_path.rglob("*.png"))) if train_path.exists() else 0
            test_count = len(list(test_path.rglob("*.png"))) if test_path.exists() else 0

            print(f"  {category}: {train_count} train, {test_count} test images")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download MVTec AD dataset")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./mvtec_ad",
        help="Root directory for dataset (default: ./mvtec_ad)"
    )

    args = parser.parse_args()

    download_mvtec_ad(args.root_dir)
