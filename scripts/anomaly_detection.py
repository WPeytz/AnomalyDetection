"""
Zero-shot and Few-shot Anomaly Detection using DINOv3

This script implements anomaly detection on MVTec AD dataset using
pretrained DINOv3 embeddings.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import json

from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
from embedding_extractor import (
    DINOv3EmbeddingExtractor,
    compute_anomaly_scores,
)


def evaluate_anomaly_detection(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Evaluate anomaly detection performance.

    Args:
        anomaly_scores: Anomaly scores (N,)
        labels: Ground truth labels (N,) - 0 for normal, 1 for anomaly

    Returns:
        Dictionary with evaluation metrics
    """
    # Compute metrics
    auroc = roc_auc_score(labels, anomaly_scores)
    ap = average_precision_score(labels, anomaly_scores)

    return {
        'auroc': float(auroc),
        'average_precision': float(ap),
    }


def visualize_anomaly_map(
    image: torch.Tensor,
    anomaly_score: float,
    label: int,
    defect_type: str,
    save_path: Path = None,
):
    """
    Visualize an image with its anomaly score.

    Args:
        image: Image tensor (C, H, W)
        anomaly_score: Anomaly score
        label: Ground truth label
        defect_type: Defect type name
        save_path: Path to save the visualization
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_np)
    ax.axis('off')

    # Add title with anomaly score
    color = 'red' if label == 1 else 'green'
    title = f"Defect: {defect_type}\nAnomaly Score: {anomaly_score:.3f}"
    ax.set_title(title, color=color, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def run_anomaly_detection(
    category: str,
    root_dir: str = "./mvtec_ad",
    model_name: str = "dinov3_vits16",
    batch_size: int = 32,
    image_size: int = 224,
    use_patches: bool = True,
    k_neighbors: int = 1,
    output_dir: str = "./results",
    visualize_samples: int = 5,
):
    """
    Run anomaly detection on MVTec AD category.

    Args:
        category: MVTec AD category name
        root_dir: Root directory of MVTec AD dataset
        model_name: DINOv3 model name
        batch_size: Batch size for processing
        image_size: Input image size
        use_patches: Whether to use patch embeddings (vs CLS token)
        k_neighbors: Number of nearest neighbors for anomaly scoring
        output_dir: Directory to save results
        visualize_samples: Number of samples to visualize
    """
    print(f"\n{'='*70}")
    print(f"Running Anomaly Detection: {category}")
    print(f"{'='*70}\n")

    # Setup output directory
    output_dir = Path(output_dir) / category
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model
    print(f"Loading model: {model_name}")
    extractor = DINOv3EmbeddingExtractor(
        model_name=model_name,
        use_huggingface=True,
    )

    # Setup transforms
    transform = get_mvtec_transforms(image_size)

    # Load training data (only normal samples)
    print(f"\nLoading training data...")
    train_dataset = MVTecADDataset(
        root=root_dir,
        category=category,
        split='train',
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Extract training embeddings
    print(f"Training samples: {len(train_dataset)}")
    train_embeddings = extractor.extract_embeddings_batch(
        train_loader,
        extract_patches=use_patches,
        extract_cls=True,
    )

    # Get normal embeddings for reference
    if use_patches:
        normal_embeddings = train_embeddings['patch_embeddings']
    else:
        normal_embeddings = train_embeddings['cls_embeddings']

    print(f"Normal embeddings shape: {normal_embeddings.shape}")

    # Load test data
    print(f"\nLoading test data...")
    test_dataset = MVTecADDataset(
        root=root_dir,
        category=category,
        split='test',
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract test embeddings
    print(f"Test samples: {len(test_dataset)}")
    test_embeddings = extractor.extract_embeddings_batch(
        test_loader,
        extract_patches=use_patches,
        extract_cls=True,
    )

    if use_patches:
        test_embed = test_embeddings['patch_embeddings']
    else:
        test_embed = test_embeddings['cls_embeddings']

    print(f"Test embeddings shape: {test_embed.shape}")

    # Compute anomaly scores
    print(f"\nComputing anomaly scores (k={k_neighbors})...")
    anomaly_scores = compute_anomaly_scores(
        test_embed,
        normal_embeddings,
        metric='cosine',
        k=k_neighbors,
    )

    # Evaluate
    print(f"\nEvaluating...")
    metrics = evaluate_anomaly_detection(
        anomaly_scores,
        test_embeddings['labels'],
    )

    print(f"\n{'='*70}")
    print(f"Results for {category}:")
    print(f"{'='*70}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print(f"{'='*70}\n")

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Visualize samples
    if visualize_samples > 0:
        print(f"\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        # Visualize top anomalies
        anomaly_indices = np.argsort(anomaly_scores)[::-1][:visualize_samples]

        for i, idx in enumerate(anomaly_indices):
            sample = test_dataset[idx]
            save_path = vis_dir / f"top_anomaly_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            visualize_anomaly_map(
                sample['image'],
                anomaly_scores[idx],
                sample['label'],
                sample['defect_type'],
                save_path,
            )

        # Visualize top normal samples
        normal_indices = np.argsort(anomaly_scores)[:visualize_samples]

        for i, idx in enumerate(normal_indices):
            sample = test_dataset[idx]
            save_path = vis_dir / f"top_normal_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            visualize_anomaly_map(
                sample['image'],
                anomaly_scores[idx],
                sample['label'],
                sample['defect_type'],
                save_path,
            )

    # Save anomaly scores
    scores_file = output_dir / "anomaly_scores.npz"
    np.savez(
        scores_file,
        scores=anomaly_scores,
        labels=test_embeddings['labels'],
    )
    print(f"Saved anomaly scores to {scores_file}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot anomaly detection on MVTec AD using DINOv3"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="bottle",
        choices=[
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'all'
        ],
        help="MVTec AD category to evaluate (or 'all' for all categories)"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="./mvtec_ad",
        help="Root directory of MVTec AD dataset"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="dinov3_vits16",
        choices=[
            'dinov3_vits16', 'dinov3_vits16plus', 'dinov3_vitb16',
            'dinov3_vitl16', 'dinov3_vith16plus', 'dinov3_vit7b16'
        ],
        help="DINOv3 model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size"
    )
    parser.add_argument(
        "--use-cls",
        action="store_true",
        help="Use CLS token instead of patch embeddings"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=1,
        help="Number of nearest neighbors for anomaly scoring"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=5,
        help="Number of samples to visualize"
    )

    args = parser.parse_args()

    # Run on all categories or single category
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ] if args.category == 'all' else [args.category]

    all_metrics = {}

    for category in categories:
        try:
            metrics = run_anomaly_detection(
                category=category,
                root_dir=args.root_dir,
                model_name=args.model_name,
                batch_size=args.batch_size,
                image_size=args.image_size,
                use_patches=not args.use_cls,
                k_neighbors=args.k_neighbors,
                output_dir=args.output_dir,
                visualize_samples=args.visualize_samples,
            )
            all_metrics[category] = metrics
        except Exception as e:
            print(f"Error processing {category}: {e}")
            continue

    # Print summary if multiple categories
    if len(categories) > 1:
        print(f"\n{'='*70}")
        print("Summary of Results:")
        print(f"{'='*70}")
        print(f"{'Category':<15} {'AUROC':<10} {'AP':<10}")
        print(f"{'-'*70}")
        for category, metrics in all_metrics.items():
            print(f"{category:<15} {metrics['auroc']:<10.4f} {metrics['average_precision']:<10.4f}")

        # Compute average
        avg_auroc = np.mean([m['auroc'] for m in all_metrics.values()])
        avg_ap = np.mean([m['average_precision'] for m in all_metrics.values()])
        print(f"{'-'*70}")
        print(f"{'Average':<15} {avg_auroc:<10.4f} {avg_ap:<10.4f}")
        print(f"{'='*70}\n")

        # Save summary
        summary_file = Path(args.output_dir) / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
