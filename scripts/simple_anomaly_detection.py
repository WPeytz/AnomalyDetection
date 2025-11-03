"""
Simple anomaly detection using torch.hub (no transformers required).

This version uses torch.hub to load DINOv3 models, avoiding the
transformers library which has threading issues on some macOS systems.
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

from mvtec_dataset import MVTecADDataset, get_mvtec_transforms


def custom_collate_fn(batch):
    """Custom collate function to handle None values in masks."""
    # Separate items by key
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    masks = [item['mask'] for item in batch]  # Keep as list, may contain None
    defect_types = [item['defect_type'] for item in batch]
    image_paths = [item['image_path'] for item in batch]

    return {
        'image': images,
        'label': labels,
        'mask': masks,
        'defect_type': defect_types,
        'image_path': image_paths,
    }


def load_dinov3_model(model_name='dinov3_vits16'):
    """Load DINOv3 model using torch.hub."""
    print(f"Loading {model_name} via torch.hub...")

    # Get repo directory (parent of scripts)
    repo_dir = Path(__file__).parent.parent

    # Try to load from local repo
    try:
        model = torch.hub.load(
            str(repo_dir),
            model_name,
            source='local',
            pretrained=True
        )
        print(f"✓ Model loaded from local repo")
    except Exception as e:
        print(f"Could not load from local repo: {e}")
        print("Downloading from torch.hub...")
        model = torch.hub.load(
            'facebookresearch/dinov3',
            model_name,
            source='github',
            pretrained=True
        )
        print(f"✓ Model downloaded and loaded")

    model.eval()
    return model


@torch.no_grad()
def extract_embeddings(model, dataloader, use_patches=True):
    """Extract embeddings using DINOv3."""
    cls_embeddings = []
    patch_embeddings = []
    labels = []

    device = next(model.parameters()).device

    print("Extracting embeddings...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        batch_labels = batch['label']

        # Forward pass
        output = model(images, is_training=False)

        # Handle different output formats
        if isinstance(output, dict):
            # DINOv3 returns dict
            cls_token = output['x_norm_clstoken']
            patch_token = output['x_norm_patchtokens']
        else:
            # Fallback: assume it's a tensor (B, N+1, D)
            cls_token = output[:, 0]
            patch_token = output[:, 1:]

        cls_embeddings.append(cls_token.cpu().numpy())

        if use_patches:
            patch_embeddings.append(patch_token.cpu().numpy())

        labels.append(batch_labels.numpy())

    result = {
        'cls_embeddings': np.concatenate(cls_embeddings, axis=0),
        'labels': np.concatenate(labels, axis=0)
    }

    if use_patches:
        result['patch_embeddings'] = np.concatenate(patch_embeddings, axis=0)

    return result


def compute_anomaly_scores(test_embed, normal_embed, k=1):
    """Compute anomaly scores based on cosine similarity."""
    # Normalize embeddings
    test_norm = test_embed / (np.linalg.norm(test_embed, axis=-1, keepdims=True) + 1e-8)
    normal_norm = normal_embed / (np.linalg.norm(normal_embed, axis=-1, keepdims=True) + 1e-8)

    if test_embed.ndim == 2:
        # CLS token embeddings: (N, D) @ (M, D).T -> (N, M)
        similarities = test_norm @ normal_norm.T
    else:
        # Patch embeddings: (N, P, D)
        N, P, D = test_embed.shape
        M = normal_embed.shape[0]

        similarities = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                # Compute pairwise similarities between patches
                patch_sim = test_norm[i] @ normal_norm[j].T  # (P, P)
                # Take max similarity for each query patch, then average
                similarities[i, j] = patch_sim.max(axis=1).mean()

    # Get top-k similarities
    if k == 1:
        max_similarities = similarities.max(axis=1)
    else:
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
        max_similarities = np.mean(
            np.take_along_axis(similarities, top_k_indices, axis=1),
            axis=1
        )

    # Convert to anomaly scores (lower similarity = higher anomaly)
    anomaly_scores = 1 - max_similarities

    return anomaly_scores


def evaluate(anomaly_scores, labels):
    """Evaluate anomaly detection performance."""
    auroc = roc_auc_score(labels, anomaly_scores)
    ap = average_precision_score(labels, anomaly_scores)

    return {
        'auroc': float(auroc),
        'average_precision': float(ap),
    }


def main():
    parser = argparse.ArgumentParser(description="Simple anomaly detection with DINOv3")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--root-dir", type=str, default="../mvtec_ad")
    parser.add_argument("--model-name", type=str, default="dinov3_vits16",
                       choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-cls", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="../results")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"Anomaly Detection: {args.category}")
    print("=" * 70 + "\n")

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model = load_dinov3_model(args.model_name)
    model = model.to(device)

    # Setup transforms and datasets
    transform = get_mvtec_transforms(224)

    print("\nLoading training data...")
    train_dataset = MVTecADDataset(
        root=args.root_dir,
        category=args.category,
        split='train',
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print(f"  Train samples: {len(train_dataset)}")

    print("\nLoading test data...")
    test_dataset = MVTecADDataset(
        root=args.root_dir,
        category=args.category,
        split='test',
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print(f"  Test samples: {len(test_dataset)}")
    print(f"    - Normal: {len(test_dataset.get_normal_samples())}")
    print(f"    - Anomaly: {len(test_dataset.get_anomaly_samples())}\n")

    # Extract embeddings
    train_embeddings = extract_embeddings(model, train_loader, use_patches=not args.use_cls)
    test_embeddings = extract_embeddings(model, test_loader, use_patches=not args.use_cls)

    # Select embedding type
    if args.use_cls:
        train_embed = train_embeddings['cls_embeddings']
        test_embed = test_embeddings['cls_embeddings']
        print("Using CLS token embeddings")
    else:
        train_embed = train_embeddings['patch_embeddings']
        test_embed = test_embeddings['patch_embeddings']
        print("Using patch embeddings")

    print(f"  Train shape: {train_embed.shape}")
    print(f"  Test shape: {test_embed.shape}")

    # Compute anomaly scores
    print(f"\nComputing anomaly scores (k={args.k_neighbors})...")
    anomaly_scores = compute_anomaly_scores(test_embed, train_embed, k=args.k_neighbors)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(anomaly_scores, test_embeddings['labels'])

    print("\n" + "=" * 70)
    print(f"Results for {args.category}:")
    print("=" * 70)
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print("=" * 70 + "\n")

    # Save results
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    scores_file = output_dir / "anomaly_scores.npz"
    np.savez(scores_file, scores=anomaly_scores, labels=test_embeddings['labels'])
    print(f"Saved anomaly scores to {scores_file}")


if __name__ == "__main__":
    main()
