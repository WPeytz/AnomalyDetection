"""
Similarity Analysis: Compare Normal vs Defective Samples

This script provides detailed similarity metric analysis between normal and
defective samples, including:
- Intra-class similarity (normal-to-normal, defect-to-defect)
- Inter-class similarity (normal-to-defect)
- Statistical analysis and visualizations
- Multiple similarity metrics (cosine, euclidean, etc.)
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
from embedding_extractor import DINOv3EmbeddingExtractor


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


def compute_pairwise_similarities(embeddings1, embeddings2, metric='cosine'):
    """
    Compute pairwise similarities between two sets of embeddings.

    Args:
        embeddings1: (N, D) or (N, P, D) numpy array
        embeddings2: (M, D) or (M, P, D) numpy array
        metric: 'cosine' or 'euclidean'

    Returns:
        similarities: (N, M) similarity matrix
    """
    if metric == 'cosine':
        # Normalize embeddings
        if embeddings1.ndim == 2:
            # CLS token embeddings
            norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-8)
            norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-8)
            similarities = norm1 @ norm2.T
        else:
            # Patch embeddings - compute average max similarity
            N, P, D = embeddings1.shape
            M = embeddings2.shape[0]

            norm1 = embeddings1 / (np.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-8)
            norm2 = embeddings2 / (np.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-8)

            similarities = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    patch_sim = norm1[i] @ norm2[j].T  # (P, P)
                    similarities[i, j] = patch_sim.max(axis=1).mean()

    elif metric == 'euclidean':
        if embeddings1.ndim == 2:
            # CLS token embeddings - convert distance to similarity
            distances = np.linalg.norm(
                embeddings1[:, None, :] - embeddings2[None, :, :],
                axis=-1
            )
            # Convert to similarity (closer = more similar)
            similarities = 1 / (1 + distances)
        else:
            # Patch embeddings
            N, P, D = embeddings1.shape
            M = embeddings2.shape[0]

            similarities = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    dists = np.linalg.norm(
                        embeddings1[i, :, None, :] - embeddings2[j, None, :, :],
                        axis=-1
                    )
                    # Convert min distance to similarity
                    similarities[i, j] = np.mean(1 / (1 + dists.min(axis=1)))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarities


def analyze_similarity_distributions(normal_embed, defect_embed, metric='cosine'):
    """
    Analyze similarity distributions for different sample pairs.

    Returns:
        Dictionary with similarity statistics
    """
    results = {}

    # 1. Normal-to-Normal similarity (intra-class)
    print("Computing normal-to-normal similarities...")
    normal_normal_sim = compute_pairwise_similarities(normal_embed, normal_embed, metric)
    # Exclude diagonal (self-similarity)
    mask = ~np.eye(normal_normal_sim.shape[0], dtype=bool)
    normal_normal_scores = normal_normal_sim[mask]

    results['normal_to_normal'] = {
        'mean': float(np.mean(normal_normal_scores)),
        'std': float(np.std(normal_normal_scores)),
        'min': float(np.min(normal_normal_scores)),
        'max': float(np.max(normal_normal_scores)),
        'median': float(np.median(normal_normal_scores)),
        'scores': normal_normal_scores,
    }

    # 2. Defect-to-Defect similarity (intra-class)
    print("Computing defect-to-defect similarities...")
    defect_defect_sim = compute_pairwise_similarities(defect_embed, defect_embed, metric)
    mask = ~np.eye(defect_defect_sim.shape[0], dtype=bool)
    defect_defect_scores = defect_defect_sim[mask]

    results['defect_to_defect'] = {
        'mean': float(np.mean(defect_defect_scores)),
        'std': float(np.std(defect_defect_scores)),
        'min': float(np.min(defect_defect_scores)),
        'max': float(np.max(defect_defect_scores)),
        'median': float(np.median(defect_defect_scores)),
        'scores': defect_defect_scores,
    }

    # 3. Normal-to-Defect similarity (inter-class)
    print("Computing normal-to-defect similarities...")
    normal_defect_sim = compute_pairwise_similarities(normal_embed, defect_embed, metric)
    normal_defect_scores = normal_defect_sim.flatten()

    results['normal_to_defect'] = {
        'mean': float(np.mean(normal_defect_scores)),
        'std': float(np.std(normal_defect_scores)),
        'min': float(np.min(normal_defect_scores)),
        'max': float(np.max(normal_defect_scores)),
        'median': float(np.median(normal_defect_scores)),
        'scores': normal_defect_scores,
    }

    # 4. Statistical tests
    print("Running statistical tests...")

    # T-test: normal-normal vs normal-defect
    t_stat, p_value = stats.ttest_ind(normal_normal_scores, normal_defect_scores)
    results['ttest_normal_vs_defect'] = {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05),
    }

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(normal_normal_scores)**2 + np.std(normal_defect_scores)**2) / 2)
    cohens_d = (np.mean(normal_normal_scores) - np.mean(normal_defect_scores)) / pooled_std
    results['cohens_d'] = float(cohens_d)

    # Separability metric
    separation = (np.mean(normal_normal_scores) - np.mean(normal_defect_scores)) / (
        np.std(normal_normal_scores) + np.std(normal_defect_scores)
    )
    results['separability'] = float(separation)

    return results


def visualize_similarity_distributions(results, save_path):
    """Create comprehensive similarity distribution visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Histogram comparison
    ax = axes[0, 0]
    ax.hist(results['normal_to_normal']['scores'], bins=50, alpha=0.6,
            label='Normal-to-Normal', color='green', density=True)
    ax.hist(results['defect_to_defect']['scores'], bins=50, alpha=0.6,
            label='Defect-to-Defect', color='orange', density=True)
    ax.hist(results['normal_to_defect']['scores'], bins=50, alpha=0.6,
            label='Normal-to-Defect', color='red', density=True)
    ax.set_xlabel('Similarity Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Similarity Score Distributions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plots
    ax = axes[0, 1]
    data_to_plot = [
        results['normal_to_normal']['scores'],
        results['defect_to_defect']['scores'],
        results['normal_to_defect']['scores'],
    ]
    bp = ax.boxplot(data_to_plot, labels=['Normal-Normal', 'Defect-Defect', 'Normal-Defect'],
                     patch_artist=True)
    colors = ['lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_title('Similarity Score Box Plots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Violin plots
    ax = axes[1, 0]
    parts = ax.violinplot(data_to_plot, positions=[1, 2, 3], showmeans=True, showmedians=True)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Normal-Normal', 'Defect-Defect', 'Normal-Defect'])
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.set_title('Similarity Score Violin Plots', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Statistical summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = [
        ['Metric', 'Normal-Normal', 'Defect-Defect', 'Normal-Defect'],
        ['Mean', f"{results['normal_to_normal']['mean']:.4f}",
         f"{results['defect_to_defect']['mean']:.4f}",
         f"{results['normal_to_defect']['mean']:.4f}"],
        ['Std', f"{results['normal_to_normal']['std']:.4f}",
         f"{results['defect_to_defect']['std']:.4f}",
         f"{results['normal_to_defect']['std']:.4f}"],
        ['Min', f"{results['normal_to_normal']['min']:.4f}",
         f"{results['defect_to_defect']['min']:.4f}",
         f"{results['normal_to_defect']['min']:.4f}"],
        ['Max', f"{results['normal_to_normal']['max']:.4f}",
         f"{results['defect_to_defect']['max']:.4f}",
         f"{results['normal_to_defect']['max']:.4f}"],
        ['', '', '', ''],
        ['Cohen\'s d', f"{results['cohens_d']:.4f}", '', ''],
        ['Separability', f"{results['separability']:.4f}", '', ''],
        ['T-test p-value', f"{results['ttest_normal_vs_defect']['p_value']:.4e}", '', ''],
    ]

    table = ax.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved similarity distribution plot to {save_path}")
    plt.close()


def visualize_embedding_space(normal_embed, defect_embed, save_path, method='tsne'):
    """Visualize embeddings in 2D using dimensionality reduction."""
    print(f"Computing {method.upper()} visualization...")

    # Flatten patch embeddings if needed
    if normal_embed.ndim == 3:
        # Average across patches
        normal_embed = normal_embed.mean(axis=1)
    if defect_embed.ndim == 3:
        defect_embed = defect_embed.mean(axis=1)

    # Combine embeddings
    all_embeddings = np.vstack([normal_embed, defect_embed])
    labels = np.concatenate([
        np.zeros(len(normal_embed)),
        np.ones(len(defect_embed))
    ])

    # Reduce dimensionality
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    else:
        raise ValueError(f"Unknown method: {method}")

    embeddings_2d = reducer.fit_transform(all_embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot normal samples
    normal_mask = labels == 0
    ax.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1],
              c='green', label='Normal', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Plot defect samples
    defect_mask = labels == 1
    ax.scatter(embeddings_2d[defect_mask, 0], embeddings_2d[defect_mask, 1],
              c='red', label='Defect', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_title(f'Embedding Space Visualization ({method.upper()})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {method.upper()} visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Similarity analysis for normal vs defective samples")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--root-dir", type=str, default="../mvtec_ad")
    parser.add_argument("--model-name", type=str, default="dinov3_vits16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-cls", action="store_true", help="Use CLS token instead of patch embeddings")
    parser.add_argument("--metric", type=str, default="cosine", choices=['cosine', 'euclidean'])
    parser.add_argument("--output-dir", type=str, default="../results")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"Similarity Analysis: {args.category}")
    print("=" * 70 + "\n")

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading model: {args.model_name}")
    from simple_anomaly_detection import load_dinov3_model, extract_embeddings
    model = load_dinov3_model(args.model_name)
    model = model.to(device)

    # Setup transforms and datasets
    transform = get_mvtec_transforms(224)

    # Load test data
    print("\nLoading test data...")
    test_dataset = MVTecADDataset(
        root=args.root_dir,
        category=args.category,
        split='test',
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=custom_collate_fn)
    print(f"Test samples: {len(test_dataset)}")

    # Extract embeddings
    print("\nExtracting embeddings...")
    test_embeddings = extract_embeddings(model, test_loader, use_patches=not args.use_cls)

    # Separate normal and defect samples
    labels = test_embeddings['labels']
    normal_mask = labels == 0
    defect_mask = labels == 1

    if args.use_cls:
        normal_embed = test_embeddings['cls_embeddings'][normal_mask]
        defect_embed = test_embeddings['cls_embeddings'][defect_mask]
        embed_type = "CLS token"
    else:
        normal_embed = test_embeddings['patch_embeddings'][normal_mask]
        defect_embed = test_embeddings['patch_embeddings'][defect_mask]
        embed_type = "Patch"

    print(f"\nUsing {embed_type} embeddings:")
    print(f"  Normal samples: {len(normal_embed)} with shape {normal_embed.shape}")
    print(f"  Defect samples: {len(defect_embed)} with shape {defect_embed.shape}")

    # Analyze similarities
    print(f"\nAnalyzing similarities using {args.metric} metric...")
    results = analyze_similarity_distributions(normal_embed, defect_embed, metric=args.metric)

    # Print summary
    print("\n" + "=" * 70)
    print("SIMILARITY ANALYSIS RESULTS")
    print("=" * 70)
    print(f"\nNormal-to-Normal Similarity:")
    print(f"  Mean: {results['normal_to_normal']['mean']:.4f} ± {results['normal_to_normal']['std']:.4f}")
    print(f"  Range: [{results['normal_to_normal']['min']:.4f}, {results['normal_to_normal']['max']:.4f}]")

    print(f"\nDefect-to-Defect Similarity:")
    print(f"  Mean: {results['defect_to_defect']['mean']:.4f} ± {results['defect_to_defect']['std']:.4f}")
    print(f"  Range: [{results['defect_to_defect']['min']:.4f}, {results['defect_to_defect']['max']:.4f}]")

    print(f"\nNormal-to-Defect Similarity:")
    print(f"  Mean: {results['normal_to_defect']['mean']:.4f} ± {results['normal_to_defect']['std']:.4f}")
    print(f"  Range: [{results['normal_to_defect']['min']:.4f}, {results['normal_to_defect']['max']:.4f}]")

    print(f"\nStatistical Significance:")
    print(f"  T-test p-value: {results['ttest_normal_vs_defect']['p_value']:.4e}")
    print(f"  Significant: {results['ttest_normal_vs_defect']['significant']}")
    print(f"  Cohen's d: {results['cohens_d']:.4f}")
    print(f"  Separability: {results['separability']:.4f}")
    print("=" * 70 + "\n")

    # Setup output directory
    output_dir = Path(args.output_dir) / args.category / "similarity_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations first (needs the scores)
    print("\nGenerating visualizations...")

    # Similarity distributions
    viz_path = output_dir / "similarity_distributions.png"
    visualize_similarity_distributions(results, viz_path)

    # Embedding space (t-SNE)
    tsne_path = output_dir / "embedding_space_tsne.png"
    visualize_embedding_space(normal_embed, defect_embed, tsne_path, method='tsne')

    # Embedding space (PCA)
    pca_path = output_dir / "embedding_space_pca.png"
    visualize_embedding_space(normal_embed, defect_embed, pca_path, method='pca')

    # Save results (remove scores arrays for JSON serialization)
    results_to_save = {k: v for k, v in results.items()}
    for key in ['normal_to_normal', 'defect_to_defect', 'normal_to_defect']:
        if 'scores' in results_to_save[key]:
            del results_to_save[key]['scores']

    results_file = output_dir / "similarity_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {results_file}")

    print("\n" + "=" * 70)
    print("Similarity analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
