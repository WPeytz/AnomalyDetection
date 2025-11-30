"""
Generate Anomaly Maps using Prompt-Based Feature Adaptation.

This script loads a trained prompt checkpoint and generates patch-level
anomaly heatmaps for test images, visualizing where defects are detected.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
from prompt_model import VisualPromptTuning, load_dinov3_for_prompting


def custom_collate_fn(batch):
    """Custom collate function to handle None values in masks."""
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    masks = [item['mask'] for item in batch]
    defect_types = [item['defect_type'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    return {
        'image': images,
        'label': labels,
        'mask': masks,
        'defect_type': defect_types,
        'image_path': image_paths,
    }


def load_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """Load trained prompts from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get modulation settings (with backwards compatibility)
    modulation_type = checkpoint.get('modulation_type', 'global')
    scaling_factor = checkpoint.get('scaling_factor', 0.5)

    # Load model with same configuration
    model = load_dinov3_for_prompting(
        model_name=checkpoint['model_name'],
        num_prompts=checkpoint['num_prompts'],
        device=device,
        modulation_type=modulation_type,
        scaling_factor=scaling_factor,
    )

    # Load trained prompts
    model.prompts.data = checkpoint['prompts'].to(device)

    # Load prompt scales if available
    if 'prompt_scales' in checkpoint:
        model.prompt_scales.data = checkpoint['prompt_scales'].to(device)

    # Load MLP weights if using learned_transform
    if modulation_type == 'learned_transform' and 'transform_mlp_state' in checkpoint:
        model.transform_mlp.load_state_dict(checkpoint['transform_mlp_state'])

    return model, checkpoint


@torch.no_grad()
def extract_patch_embeddings(model, dataloader, device='cpu'):
    """Extract patch embeddings using the prompt-tuned model."""
    all_patch_embeddings = []
    all_labels = []
    all_defect_types = []
    all_image_paths = []
    all_images = []

    print("Extracting patch embeddings...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)

        # Forward pass through prompt-tuned model
        output = model(images, return_cls=False, return_patches=True)

        all_patch_embeddings.append(output['patch_embeddings'].cpu().numpy())
        all_labels.extend(batch['label'].numpy().tolist())
        all_defect_types.extend(batch['defect_type'])
        all_image_paths.extend(batch['image_path'])
        all_images.extend([img.cpu() for img in images])

    return {
        'patch_embeddings': np.concatenate(all_patch_embeddings, axis=0),
        'labels': np.array(all_labels),
        'defect_types': all_defect_types,
        'image_paths': all_image_paths,
        'images': all_images,
    }


def compute_patch_anomaly_scores(test_embeddings, normal_embeddings):
    """
    Compute patch-level anomaly scores by comparing test patches to normal patches.

    Args:
        test_embeddings: Test patch embeddings [N_test, P, D]
        normal_embeddings: Normal patch embeddings [N_normal, P, D]

    Returns:
        Patch-level anomaly scores [N_test, P]
    """
    N_test, P, D = test_embeddings.shape
    N_normal = normal_embeddings.shape[0]

    # Normalize embeddings
    test_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=-1, keepdims=True) + 1e-8)
    normal_norm = normal_embeddings / (np.linalg.norm(normal_embeddings, axis=-1, keepdims=True) + 1e-8)

    # Flatten normal patches: [N_normal, P, D] -> [N_normal * P, D]
    normal_flat = normal_norm.reshape(-1, D)

    patch_anomaly_scores = np.zeros((N_test, P))

    print("Computing patch-level anomaly scores...")
    for i in tqdm(range(N_test)):
        for p in range(P):
            # Get this patch embedding: [D]
            test_patch = test_norm[i, p]

            # Compute similarity to all normal patches: [N_normal * P]
            similarities = test_patch @ normal_flat.T

            # Max similarity (nearest neighbor)
            max_sim = similarities.max()

            # Anomaly score = 1 - similarity
            patch_anomaly_scores[i, p] = 1 - max_sim

    return patch_anomaly_scores


def create_anomaly_heatmap(patch_scores, image_size=224, grid_size=14):
    """
    Convert patch-level scores to a spatial anomaly heatmap.

    Args:
        patch_scores: Patch anomaly scores [P] where P = grid_size^2
        image_size: Original image size (224)
        grid_size: Patch grid size (14 for DINOv3 with 224 images)

    Returns:
        Heatmap of shape [image_size, image_size]
    """
    expected_patches = grid_size * grid_size  # 196

    # Handle register tokens if present
    if len(patch_scores) > expected_patches:
        n_registers = len(patch_scores) - expected_patches
        patch_scores = patch_scores[n_registers:n_registers + expected_patches]

    # Reshape to grid
    if len(patch_scores) == expected_patches:
        heatmap = patch_scores.reshape(grid_size, grid_size)
    else:
        # Pad if needed
        padded = np.zeros(expected_patches)
        padded[:len(patch_scores)] = patch_scores
        heatmap = padded.reshape(grid_size, grid_size)

    # Upsample to image size
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC)

    return heatmap_resized


def visualize_anomaly_map(
    image_tensor,
    anomaly_heatmap,
    anomaly_score,
    label,
    defect_type,
    save_path=None,
):
    """
    Visualize the anomaly heatmap overlaid on the original image.

    Args:
        image_tensor: Image tensor [C, H, W]
        anomaly_heatmap: 2D anomaly heatmap [H, W]
        anomaly_score: Image-level anomaly score
        label: Ground truth label (0=normal, 1=anomaly)
        defect_type: Type of defect
        save_path: Path to save visualization
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image_tensor * std + mean
    image = torch.clamp(image, 0, 1)
    image_np = image.permute(1, 2, 0).numpy()

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # Anomaly heatmap only
    im1 = axes[1].imshow(anomaly_heatmap, cmap='hot')
    axes[1].set_title('Anomaly Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(image_np)
    axes[2].imshow(anomaly_heatmap, cmap='hot', alpha=0.6)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    # Title with score and label
    status = "ANOMALY" if label == 1 else "NORMAL"
    color = 'red' if label == 1 else 'green'
    fig.suptitle(
        f"{defect_type} | Score: {anomaly_score:.4f} | {status}",
        fontsize=14, fontweight='bold', color=color
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_anomaly_maps(
    category: str,
    checkpoint_path: str,
    root_dir: str = "mvtec_ad",
    output_dir: str = None,
    batch_size: int = 16,
    num_samples: int = 10,
    device: str = 'cpu',
):
    """
    Generate and visualize anomaly maps for a category using prompt-based adaptation.

    Args:
        category: MVTec AD category
        checkpoint_path: Path to trained prompt checkpoint
        root_dir: MVTec AD dataset root directory
        output_dir: Output directory for visualizations
        batch_size: Batch size for processing
        num_samples: Number of samples to visualize
        device: Device to use
    """
    print("=" * 70)
    print(f"Generating Anomaly Maps: {category}")
    print(f"Using checkpoint: {checkpoint_path}")
    print("=" * 70)

    # Setup output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / "anomaly_maps" / category
    else:
        output_dir = Path(output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading prompt-tuned model...")
    model, checkpoint = load_checkpoint(Path(checkpoint_path), device)
    model.eval()

    print(f"  Category trained on: {checkpoint.get('category', 'unknown')}")
    print(f"  Modulation type: {checkpoint.get('modulation_type', 'global')}")
    print(f"  Num prompts: {checkpoint.get('num_prompts', 10)}")

    # Setup transforms
    transform = get_mvtec_transforms(224)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = MVTecADDataset(
        root=root_dir,
        category=category,
        split='train',
        transform=transform,
    )
    test_dataset = MVTecADDataset(
        root=root_dir,
        category=category,
        split='test',
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    print(f"  Train samples (normal): {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Extract embeddings
    print("\nExtracting embeddings from training data (normal)...")
    train_data = extract_patch_embeddings(model, train_loader, device)
    normal_embeddings = train_data['patch_embeddings']

    print("\nExtracting embeddings from test data...")
    test_data = extract_patch_embeddings(model, test_loader, device)
    test_embeddings = test_data['patch_embeddings']

    print(f"\nNormal embeddings shape: {normal_embeddings.shape}")
    print(f"Test embeddings shape: {test_embeddings.shape}")

    # Compute patch-level anomaly scores
    patch_anomaly_scores = compute_patch_anomaly_scores(test_embeddings, normal_embeddings)

    # Image-level scores (max of patch scores)
    image_anomaly_scores = patch_anomaly_scores.max(axis=1)

    print(f"\nPatch anomaly scores shape: {patch_anomaly_scores.shape}")
    print(f"Image anomaly scores range: [{image_anomaly_scores.min():.4f}, {image_anomaly_scores.max():.4f}]")

    # Get indices of top anomalies and top normals
    anomaly_indices = np.where(test_data['labels'] == 1)[0]
    normal_indices = np.where(test_data['labels'] == 0)[0]

    # Sort by anomaly score
    sorted_anomaly_idx = anomaly_indices[np.argsort(image_anomaly_scores[anomaly_indices])[::-1]]
    sorted_normal_idx = normal_indices[np.argsort(image_anomaly_scores[normal_indices])[::-1]]

    # Visualize top anomalies
    print(f"\nGenerating visualizations...")
    print(f"  Saving to: {output_dir}")

    n_anomalies = min(num_samples, len(sorted_anomaly_idx))
    n_normals = min(num_samples, len(sorted_normal_idx))

    print(f"\n  Visualizing top {n_anomalies} anomalies...")
    for i, idx in enumerate(sorted_anomaly_idx[:n_anomalies]):
        heatmap = create_anomaly_heatmap(patch_anomaly_scores[idx])

        # Normalize heatmap for visualization
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        save_path = output_dir / f"anomaly_{i+1:02d}_{test_data['defect_types'][idx]}_score_{image_anomaly_scores[idx]:.3f}.png"

        visualize_anomaly_map(
            test_data['images'][idx],
            heatmap_norm,
            image_anomaly_scores[idx],
            test_data['labels'][idx],
            test_data['defect_types'][idx],
            save_path,
        )
        print(f"    Saved: {save_path.name}")

    print(f"\n  Visualizing top {n_normals} normal samples (by score)...")
    for i, idx in enumerate(sorted_normal_idx[:n_normals]):
        heatmap = create_anomaly_heatmap(patch_anomaly_scores[idx])

        # Normalize heatmap for visualization
        heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        save_path = output_dir / f"normal_{i+1:02d}_score_{image_anomaly_scores[idx]:.3f}.png"

        visualize_anomaly_map(
            test_data['images'][idx],
            heatmap_norm,
            image_anomaly_scores[idx],
            test_data['labels'][idx],
            test_data['defect_types'][idx],
            save_path,
        )
        print(f"    Saved: {save_path.name}")

    # Compute and print AUROC
    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(test_data['labels'], image_anomaly_scores)
    ap = average_precision_score(test_data['labels'], image_anomaly_scores)

    print(f"\n{'=' * 70}")
    print(f"Results for {category}:")
    print(f"  AUROC: {auroc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"{'=' * 70}")

    print(f"\nAll visualizations saved to: {output_dir}")

    return {
        'auroc': auroc,
        'average_precision': ap,
        'output_dir': str(output_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate anomaly maps using prompt-based feature adaptation"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="MVTec AD category (e.g., screw, bottle, cable)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained prompt checkpoint"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "mvtec_ad"),
        help="Root directory of MVTec AD dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to visualize"
    )

    args = parser.parse_args()

    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")

    # Generate anomaly maps
    generate_anomaly_maps(
        category=args.category,
        checkpoint_path=args.checkpoint,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=device,
    )


if __name__ == "__main__":
    main()
