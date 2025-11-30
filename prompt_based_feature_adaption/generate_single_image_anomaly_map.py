"""
Generate Anomaly Maps for specific images using Prompt-Based Feature Adaptation.

Usage:
    python generate_single_image_anomaly_map.py --image <path> --category <category>
    python generate_single_image_anomaly_map.py --image <path> --checkpoint <checkpoint_path>

Examples:
    python generate_single_image_anomaly_map.py \
        --image mvtec_ad/bottle/test/contamination/007.png \
        --category bottle

    python generate_single_image_anomaly_map.py \
        --image mvtec_ad/hazelnut/test/crack/007.png \
        --checkpoint checkpoints/hazelnut_prompts.pt
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

# Add paths
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from prompt_model import VisualPromptTuning, load_dinov3_for_prompting


def load_checkpoint(checkpoint_path: Path, device: str = 'cpu'):
    """Load trained prompts from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    modulation_type = checkpoint.get('modulation_type', 'global')
    scaling_factor = checkpoint.get('scaling_factor', 0.5)

    model = load_dinov3_for_prompting(
        model_name=checkpoint['model_name'],
        num_prompts=checkpoint['num_prompts'],
        device=device,
        modulation_type=modulation_type,
        scaling_factor=scaling_factor,
    )

    model.prompts.data = checkpoint['prompts'].to(device)

    if 'prompt_scales' in checkpoint:
        model.prompt_scales.data = checkpoint['prompt_scales'].to(device)

    if modulation_type == 'learned_transform' and 'transform_mlp_state' in checkpoint:
        model.transform_mlp.load_state_dict(checkpoint['transform_mlp_state'])

    return model, checkpoint


def get_transforms():
    """Get transforms for input images."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_normal_embeddings(model, category: str, root_dir: str, device: str):
    """Load and compute embeddings from normal training images."""
    from mvtec_dataset import MVTecADDataset, get_mvtec_transforms

    transform = get_mvtec_transforms(224)
    train_dataset = MVTecADDataset(
        root=root_dir,
        category=category,
        split='train',
        transform=transform,
    )

    all_embeddings = []
    print(f"Extracting embeddings from {len(train_dataset)} normal training images...")

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        image = sample['image'].unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image, return_cls=False, return_patches=True)
            all_embeddings.append(output['patch_embeddings'].cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def compute_patch_anomaly_scores(test_embedding: np.ndarray, normal_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute patch-level anomaly scores.

    Args:
        test_embedding: Test patch embeddings [P, D]
        normal_embeddings: Normal patch embeddings [N, P, D]

    Returns:
        Patch anomaly scores [P]
    """
    P, D = test_embedding.shape

    test_norm = test_embedding / (np.linalg.norm(test_embedding, axis=-1, keepdims=True) + 1e-8)
    normal_norm = normal_embeddings / (np.linalg.norm(normal_embeddings, axis=-1, keepdims=True) + 1e-8)

    normal_flat = normal_norm.reshape(-1, D)

    patch_anomaly_scores = np.zeros(P)

    for p in range(P):
        test_patch = test_norm[p]
        similarities = test_patch @ normal_flat.T
        max_sim = np.nanmax(similarities)  # Use nanmax to handle any NaN values
        patch_anomaly_scores[p] = 1 - max_sim

    return patch_anomaly_scores


def create_anomaly_heatmap(patch_scores: np.ndarray, image_size: int = 224, grid_size: int = 14) -> np.ndarray:
    """
    Convert patch scores to heatmap.

    Args:
        patch_scores: Patch anomaly scores [P]
        image_size: Output image size
        grid_size: Patch grid size (14 for DINOv3 with 224 images)

    Returns:
        Heatmap of shape [image_size, image_size]
    """
    expected_patches = grid_size * grid_size

    if len(patch_scores) > expected_patches:
        n_registers = len(patch_scores) - expected_patches
        patch_scores = patch_scores[n_registers:n_registers + expected_patches]

    if len(patch_scores) == expected_patches:
        heatmap = patch_scores.reshape(grid_size, grid_size)
    else:
        padded = np.zeros(expected_patches)
        padded[:len(patch_scores)] = patch_scores
        heatmap = padded.reshape(grid_size, grid_size)

    heatmap_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return heatmap_resized


def visualize_anomaly_map(
    image_path: str,
    heatmap: np.ndarray,
    anomaly_score: float,
    save_path: Path,
    category: str = None,
    defect_type: str = None,
):
    """
    Visualize anomaly map for a single image.

    Args:
        image_path: Path to the original image
        heatmap: 2D anomaly heatmap
        anomaly_score: Image-level anomaly score
        save_path: Path to save visualization
        category: Category name (optional, extracted from path if not provided)
        defect_type: Defect type (optional, extracted from path if not provided)
    """
    # Load original image
    original = Image.open(image_path).convert('RGB')
    original = original.resize((224, 224))
    original_np = np.array(original) / 255.0

    # Normalize heatmap
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Try to extract category and defect_type from path if not provided
    if category is None or defect_type is None:
        try:
            parts = Path(image_path).parts
            if category is None:
                category = parts[-4]
            if defect_type is None:
                defect_type = parts[-2]
        except (IndexError, TypeError):
            category = category or "unknown"
            defect_type = defect_type or "unknown"

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # Anomaly heatmap only
    im1 = axes[1].imshow(heatmap_norm, cmap='hot')
    axes[1].set_title('Anomaly Heatmap', fontsize=12)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay
    axes[2].imshow(original_np)
    axes[2].imshow(heatmap_norm, cmap='hot', alpha=0.6)
    axes[2].set_title('Overlay', fontsize=12)
    axes[2].axis('off')

    fig.suptitle(
        f"{category} / {defect_type} | Anomaly Score: {anomaly_score:.4f}",
        fontsize=14, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def generate_anomaly_map(
    image_path: str,
    checkpoint_path: str = None,
    category: str = None,
    root_dir: str = None,
    output_dir: str = None,
    device: str = None,
):
    """
    Generate anomaly map for a single image.

    Args:
        image_path: Path to the test image
        checkpoint_path: Path to trained prompt checkpoint (optional if category provided)
        category: MVTec AD category (optional if checkpoint provided)
        root_dir: MVTec AD dataset root directory
        output_dir: Output directory for visualization
        device: Device to use

    Returns:
        Dictionary with anomaly score and output path
    """
    # Setup device
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    # Setup paths
    script_dir = Path(__file__).parent
    if root_dir is None:
        root_dir = str(script_dir.parent / 'mvtec_ad')

    # Determine checkpoint path
    if checkpoint_path is None:
        if category is None:
            # Try to infer category from image path
            try:
                parts = Path(image_path).parts
                category = parts[-4]
                print(f"Inferred category from path: {category}")
            except (IndexError, TypeError):
                raise ValueError("Must provide either --checkpoint or --category")

        checkpoint_path = script_dir / 'checkpoints' / f'{category}_prompts.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_path = Path(checkpoint_path)

    # Infer category from checkpoint if not provided
    if category is None:
        category = checkpoint_path.stem.replace('_prompts', '')

    # Setup output directory
    if output_dir is None:
        output_dir = script_dir / 'single_image_results'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Processing: {image_path}")
    print(f"Category: {category}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*70}")

    # Load model
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()

    print(f"Loaded checkpoint with modulation type: {checkpoint.get('modulation_type', 'global')}")

    # Get normal embeddings
    normal_embeddings = load_normal_embeddings(model, category, root_dir, device)
    print(f"Normal embeddings shape: {normal_embeddings.shape}")

    # Load and process test image
    transform = get_transforms()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get test embeddings
    with torch.no_grad():
        output = model(image_tensor, return_cls=False, return_patches=True)
        test_embedding = output['patch_embeddings'].cpu().numpy()[0]  # [P, D]

    print(f"Test embedding shape: {test_embedding.shape}")

    # Compute anomaly scores
    patch_scores = compute_patch_anomaly_scores(test_embedding, normal_embeddings)
    image_score = float(patch_scores.max())

    print(f"Patch scores range: [{patch_scores.min():.4f}, {patch_scores.max():.4f}]")
    print(f"Image anomaly score: {image_score:.4f}")

    # Create heatmap
    heatmap = create_anomaly_heatmap(patch_scores)

    # Save visualization
    image_name = Path(image_path).stem
    try:
        defect_type = Path(image_path).parts[-2]
    except (IndexError, TypeError):
        defect_type = "unknown"

    save_path = output_dir / f"{category}_{defect_type}_{image_name}_anomaly_map.png"

    visualize_anomaly_map(
        image_path,
        heatmap,
        image_score,
        save_path,
        category=category,
        defect_type=defect_type,
    )

    return {
        'anomaly_score': image_score,
        'output_path': str(save_path),
        'patch_scores': patch_scores,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate anomaly map for a single image using prompt-based feature adaptation"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the test image"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained prompt checkpoint (optional if --category provided)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="MVTec AD category (e.g., bottle, hazelnut). If not provided, will be inferred from image path or checkpoint name."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=None,
        help="Root directory of MVTec AD dataset (default: ../mvtec_ad)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualization (default: ./single_image_results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cpu', 'cuda', 'mps'],
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.checkpoint is None and args.category is None:
        # Try to infer from image path
        try:
            parts = Path(args.image).parts
            _ = parts[-4]  # This will fail if path doesn't match expected structure
        except (IndexError, TypeError):
            parser.error("Must provide either --checkpoint or --category (or use MVTec AD path structure)")

    result = generate_anomaly_map(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        category=args.category,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        device=args.device,
    )

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"  Output: {result['output_path']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
