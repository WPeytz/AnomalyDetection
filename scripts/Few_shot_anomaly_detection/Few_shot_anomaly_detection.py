"""
Few-shot anomaly detection using DINOv3.

This script performs anomaly detection using only a few normal samples
for training, making it suitable for scenarios with limited normal data.
"""

import argparse
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import random

# Import from local directory
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


def compute_anomaly_scores(test_embed, normal_embed, k=1, metric='cosine'):
    """
    Compute anomaly scores using different distance metrics.
    
    Args:
        test_embed: Test embeddings (N, D) or (N, P, D)
        normal_embed: Normal embeddings (M, D) or (M, P, D)
        k: Number of nearest neighbors to consider
        metric: Distance metric ('cosine', 'euclidean', 'knn')
        
    Returns:
        Anomaly scores (N,) - higher means more anomalous
    """
    if metric == 'cosine':
        # Cosine similarity-based scoring
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
        
    elif metric == 'euclidean':
        # Euclidean distance-based scoring
        if test_embed.ndim == 2:
            # CLS token embeddings: compute pairwise distances
            # (N, D) and (M, D) -> (N, M)
            distances = np.linalg.norm(
                test_embed[:, None, :] - normal_embed[None, :, :],
                axis=-1
            )
        else:
            # Patch embeddings: (N, P, D)
            N, P, D = test_embed.shape
            M = normal_embed.shape[0]
            
            distances = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    # Compute pairwise distances between patches
                    patch_dist = np.linalg.norm(
                        test_embed[i, :, None, :] - normal_embed[j, None, :, :],
                        axis=-1
                    )  # (P, P)
                    # Take min distance for each query patch, then average
                    distances[i, j] = patch_dist.min(axis=1).mean()
        
        # Get k nearest neighbors (smallest distances)
        if k == 1:
            min_distances = distances.min(axis=1)
        else:
            top_k_indices = np.argsort(distances, axis=1)[:, :k]
            min_distances = np.mean(
                np.take_along_axis(distances, top_k_indices, axis=1),
                axis=1
            )
        
        # Anomaly score is the distance (larger distance = more anomalous)
        # Normalize by mean distance for better scale
        anomaly_scores = min_distances
        
    elif metric == 'knn':
        # k-NN distance-based scoring (average of k nearest neighbors)
        if test_embed.ndim == 2:
            # CLS token embeddings
            distances = np.linalg.norm(
                test_embed[:, None, :] - normal_embed[None, :, :],
                axis=-1
            )
        else:
            # Patch embeddings: (N, P, D)
            N, P, D = test_embed.shape
            M = normal_embed.shape[0]
            
            distances = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    # Compute pairwise distances between patches
                    patch_dist = np.linalg.norm(
                        test_embed[i, :, None, :] - normal_embed[j, None, :, :],
                        axis=-1
                    )  # (P, P)
                    # Take min distance for each query patch, then average
                    distances[i, j] = patch_dist.min(axis=1).mean()
        
        # Sort distances and take mean of k nearest neighbors
        sorted_distances = np.sort(distances, axis=1)
        knn_distances = sorted_distances[:, :k].mean(axis=1)
        
        # Anomaly score is the k-NN distance
        anomaly_scores = knn_distances
    
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'cosine', 'euclidean', 'knn'")

    return anomaly_scores


def evaluate(anomaly_scores, labels):
    """Evaluate anomaly detection performance."""
    auroc = roc_auc_score(labels, anomaly_scores)
    ap = average_precision_score(labels, anomaly_scores)

    return {
        'auroc': float(auroc),
        'average_precision': float(ap),
    }


def create_few_shot_dataset(dataset, n_shots, seed=42):
    """Create a few-shot dataset by sampling n_shots normal samples."""
    random.seed(seed)
    np.random.seed(seed)
    
    # Get indices of normal samples
    normal_indices = dataset.get_normal_samples()
    
    if len(normal_indices) < n_shots:
        print(f"Warning: Only {len(normal_indices)} normal samples available, using all of them.")
        selected_indices = normal_indices
    else:
        # Randomly sample n_shots normal samples
        selected_indices = random.sample(normal_indices, n_shots)
    
    print(f"Selected {len(selected_indices)} normal samples for few-shot training")
    return Subset(dataset, selected_indices)



def main():
    parser = argparse.ArgumentParser(description="Few-shot anomaly detection with DINOv3")
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument("--root-dir", type=str, default="mvtec_ad")
    parser.add_argument("--model-name", type=str, default="dinov3_vits16",
                       choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-cls", action="store_true")
    parser.add_argument("--k-neighbors", type=int, default=1)
    parser.add_argument("--metric", type=str, default="cosine",
                       choices=['cosine', 'euclidean', 'knn'],
                       help="Distance metric for anomaly scoring")
    parser.add_argument("--n-shots", type=int, default=10, help="Number of normal samples for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, default="../results")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"Few-Shot Anomaly Detection: {args.category} ({args.n_shots} shots)")
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
    full_train_dataset = MVTecADDataset(
        root=args.root_dir,
        category=args.category,
        split='train',
        transform=transform,
    )
    
    # Create few-shot dataset
    train_dataset = create_few_shot_dataset(full_train_dataset, args.n_shots, args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print(f"  Total train samples available: {len(full_train_dataset)}")
    print(f"  Few-shot train samples used: {len(train_dataset)}")

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

    # Compute anomaly scores for all metrics with appropriate k values
    print(f"\nComputing anomaly scores for all metrics...")
    # Cosine: k can vary, use user's choice
    k_cosine = args.k_neighbors
    print(f"  Cosine similarity (k={k_cosine})...")
    anomaly_scores_cosine = compute_anomaly_scores(test_embed, train_embed, k=k_cosine, metric='cosine')
    
    # Euclidean: k=1 for nearest neighbor distance
    k_euclidean = 1
    print(f"  Euclidean distance (k={k_euclidean})...")
    anomaly_scores_euclidean = compute_anomaly_scores(test_embed, train_embed, k=k_euclidean, metric='euclidean')
    
    # k-NN: k=5 for robust averaging (adjustable based on n_shots)
    k_knn = min(5, max(1, len(train_dataset) // 2))  # Use k=5 or half of training samples, whichever is smaller
    print(f"  k-NN distance (k={k_knn})...")
    anomaly_scores_knn = compute_anomaly_scores(test_embed, train_embed, k=k_knn, metric='knn')
    
    # Use the selected metric for primary evaluation
    if args.metric == 'cosine':
        anomaly_scores = anomaly_scores_cosine
    elif args.metric == 'euclidean':
        anomaly_scores = anomaly_scores_euclidean
    else:  # knn
        anomaly_scores = anomaly_scores_knn
    
    print(f"\nUsing {args.metric} metric for primary evaluation")

    # Evaluate all metrics
    print("\nEvaluating all metrics...")
    metrics_cosine = evaluate(anomaly_scores_cosine, test_embeddings['labels'])
    metrics_euclidean = evaluate(anomaly_scores_euclidean, test_embeddings['labels'])
    metrics_knn = evaluate(anomaly_scores_knn, test_embeddings['labels'])
    
    # Primary metric based on user selection
    metrics = {
        'cosine': {**metrics_cosine, 'k': k_cosine},
        'euclidean': {**metrics_euclidean, 'k': k_euclidean},
        'knn': {**metrics_knn, 'k': k_knn},
        'primary_metric': args.metric,
    }

    print("\n" + "=" * 70)
    print(f"Few-Shot Results for {args.category} ({args.n_shots} shots):")
    print("=" * 70)
    print(f"  Training samples used: {len(train_dataset)}")
    print(f"\n  Cosine Similarity (k={k_cosine}):")
    print(f"    AUROC: {metrics_cosine['auroc']:.4f}")
    print(f"    Average Precision: {metrics_cosine['average_precision']:.4f}")
    print(f"\n  Euclidean Distance (k={k_euclidean}):")
    print(f"    AUROC: {metrics_euclidean['auroc']:.4f}")
    print(f"    Average Precision: {metrics_euclidean['average_precision']:.4f}")
    print(f"\n  k-NN Distance (k={k_knn}):")
    print(f"    AUROC: {metrics_knn['auroc']:.4f}")
    print(f"    Average Precision: {metrics_knn['average_precision']:.4f}")
    print("=" * 70 + "\n")

    # Save results with few-shot information in the same directory as this script
    script_dir = Path(__file__).parent
    output_dir = script_dir / "results" / f"{args.category}_fewshot_{args.n_shots}"
    print(f"Saving results to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add few-shot metadata to metrics
    metrics['n_shots'] = args.n_shots
    metrics['seed'] = args.seed
    metrics['k_neighbors'] = args.k_neighbors
    metrics['total_train_samples'] = len(full_train_dataset)
    metrics['used_train_samples'] = len(train_dataset)
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    scores_file = output_dir / "anomaly_scores.npz"
    np.savez(
        scores_file, 
        scores_cosine=anomaly_scores_cosine,
        scores_euclidean=anomaly_scores_euclidean,
        scores_knn=anomaly_scores_knn,
        labels=test_embeddings['labels'], 
        n_shots=args.n_shots, 
        seed=args.seed,
        k_cosine=k_cosine,
        k_euclidean=k_euclidean,
        k_knn=k_knn,
    )
    print(f"Saved anomaly scores to {scores_file}")


if __name__ == "__main__":
    main()
