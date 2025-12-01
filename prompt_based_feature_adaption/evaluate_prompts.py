"""
Evaluation script for prompt-tuned models.

This script evaluates prompt-tuned DINOv3 models on MVTec AD test sets
and compares performance with zero-shot baseline.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / 'scripts'))

from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
from prompt_model import load_dinov3_for_prompting


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


@torch.no_grad()
def extract_embeddings(model, dataloader, use_patches=True, device='cuda'):
    """Extract embeddings using the model."""
    cls_embeddings = []
    patch_embeddings = []
    labels = []

    print("Extracting embeddings...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        batch_labels = batch['label']

        # Forward pass
        output = model(images, return_cls=not use_patches, return_patches=use_patches)

        if use_patches:
            patch_embeddings.append(output['patch_embeddings'].cpu().numpy())
        else:
            cls_embeddings.append(output['cls_embeddings'].cpu().numpy())

        labels.append(batch_labels.numpy())

    result = {'labels': np.concatenate(labels, axis=0)}

    if use_patches:
        result['patch_embeddings'] = np.concatenate(patch_embeddings, axis=0)
    else:
        result['cls_embeddings'] = np.concatenate(cls_embeddings, axis=0)

    return result


def compute_anomaly_scores(test_embed, normal_embed, k=1):
    """Compute anomaly scores based on cosine similarity."""
    # Normalize embeddings
    test_norm = test_embed / (np.linalg.norm(test_embed, axis=-1, keepdims=True) + 1e-8)
    normal_norm = normal_embed / (np.linalg.norm(normal_embed, axis=-1, keepdims=True) + 1e-8)

    if test_embed.ndim == 2:
        # CLS token embeddings
        similarities = test_norm @ normal_norm.T
    else:
        # Patch embeddings
        N, P, D = test_embed.shape
        M = normal_embed.shape[0]

        similarities = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                patch_sim = test_norm[i] @ normal_norm[j].T
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

    # Convert to anomaly scores
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


def load_checkpoint(checkpoint_path: Path, device: str = 'cuda'):
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


def compare_with_baseline(
    category: str,
    checkpoint_path: Path,
    root_dir: str = "mvtec_ad",
    batch_size: int = 32,
    k_neighbors: int = 1,
    device: str = 'cuda',
):
    """
    Compare prompt-tuned model with zero-shot baseline.

    Args:
        category: MVTec AD category
        checkpoint_path: Path to trained prompt checkpoint
        root_dir: Root directory of MVTec AD dataset
        batch_size: Batch size for evaluation
        k_neighbors: Number of nearest neighbors
        device: Device to use

    Returns:
        Dictionary with comparison results
    """
    print("=" * 70)
    print(f"Evaluation: {category}")
    print("=" * 70)

    # Setup datasets
    transform = get_mvtec_transforms(224)

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

    print(f"\nDataset: {category}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"    - Normal: {len(test_dataset.get_normal_samples())}")
    print(f"    - Anomaly: {len(test_dataset.get_anomaly_samples())}\n")

    # Load prompt-tuned model
    print("Loading prompt-tuned model...")
    prompt_model, checkpoint = load_checkpoint(checkpoint_path, device)
    prompt_model.eval()

    # Load zero-shot baseline (just the DINOv3 model without prompts)
    print("Loading zero-shot baseline...")
    print(f"Loading {checkpoint['model_name']}...")

    repo_dir = Path(__file__).parent.parent
    try:
        baseline_dinov3 = torch.hub.load(
            str(repo_dir),
            checkpoint['model_name'],
            source='local',
            pretrained=True
        )
    except Exception as e:
        print(f"Loading from local failed: {e}")
        print("Downloading from torch.hub...")
        baseline_dinov3 = torch.hub.load(
            'facebookresearch/dinov3',
            checkpoint['model_name'],
            source='github',
            pretrained=True
        )

    baseline_dinov3 = baseline_dinov3.to(device)
    baseline_dinov3.eval()

    # Wrap in a simple container for consistent interface
    class BaselineModel(nn.Module):
        def __init__(self, dinov3):
            super().__init__()
            self.dinov3 = dinov3

        def forward(self, images, return_cls=True, return_patches=True):
            with torch.no_grad():
                if hasattr(self.dinov3, 'forward_features'):
                    output = self.dinov3.forward_features(images)
                else:
                    output = self.dinov3(images, is_training=False)

            result = {}
            if isinstance(output, dict):
                if return_cls and 'x_norm_clstoken' in output:
                    result['cls_embeddings'] = output['x_norm_clstoken']
                if return_patches and 'x_norm_patchtokens' in output:
                    result['patch_embeddings'] = output['x_norm_patchtokens']
            return result

    baseline_model = BaselineModel(baseline_dinov3).to(device)
    print("✓ Zero-shot baseline loaded")

    # Determine embedding type
    use_patches = not checkpoint['args'].get('use_cls', False)
    embed_key = 'patch_embeddings' if use_patches else 'cls_embeddings'

    # Evaluate prompt-tuned model
    print("\n" + "=" * 70)
    print("PROMPT-TUNED MODEL")
    print("=" * 70)
    train_embed_prompt = extract_embeddings(prompt_model, train_loader, use_patches, device)
    test_embed_prompt = extract_embeddings(prompt_model, test_loader, use_patches, device)

    anomaly_scores_prompt = compute_anomaly_scores(
        test_embed_prompt[embed_key],
        train_embed_prompt[embed_key],
        k=k_neighbors
    )
    metrics_prompt = evaluate(anomaly_scores_prompt, test_embed_prompt['labels'])

    print("\nPrompt-Tuned Results:")
    print(f"  AUROC: {metrics_prompt['auroc']:.4f}")
    print(f"  Average Precision: {metrics_prompt['average_precision']:.4f}")

    # Evaluate zero-shot baseline
    print("\n" + "=" * 70)
    print("ZERO-SHOT BASELINE")
    print("=" * 70)
    train_embed_baseline = extract_embeddings(baseline_model, train_loader, use_patches, device)
    test_embed_baseline = extract_embeddings(baseline_model, test_loader, use_patches, device)

    anomaly_scores_baseline = compute_anomaly_scores(
        test_embed_baseline[embed_key],
        train_embed_baseline[embed_key],
        k=k_neighbors
    )
    metrics_baseline = evaluate(anomaly_scores_baseline, test_embed_baseline['labels'])

    print("\nZero-Shot Results:")
    print(f"  AUROC: {metrics_baseline['auroc']:.4f}")
    print(f"  Average Precision: {metrics_baseline['average_precision']:.4f}")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    auroc_improvement = metrics_prompt['auroc'] - metrics_baseline['auroc']
    ap_improvement = metrics_prompt['average_precision'] - metrics_baseline['average_precision']

    print(f"AUROC Improvement: {auroc_improvement:+.4f} ({auroc_improvement/metrics_baseline['auroc']*100:+.2f}%)")
    print(f"AP Improvement: {ap_improvement:+.4f} ({ap_improvement/metrics_baseline['average_precision']*100:+.2f}%)")
    print("=" * 70)

    results = {
        'category': category,
        'prompt_tuned': metrics_prompt,
        'zero_shot': metrics_baseline,
        'improvement': {
            'auroc': float(auroc_improvement),
            'auroc_relative': float(auroc_improvement / metrics_baseline['auroc'] * 100),
            'average_precision': float(ap_improvement),
            'ap_relative': float(ap_improvement / metrics_baseline['average_precision'] * 100),
        },
        'config': checkpoint['args'],
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate prompt-tuned DINOv3 model")
    parser.add_argument("--category", type=str, required=True,
                       help="MVTec AD category")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to prompt checkpoint")
    parser.add_argument("--root-dir", type=str, default="mvtec_ad",
                       help="Root directory of MVTec AD dataset")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--k-neighbors", type=int, default=1,
                       help="Number of nearest neighbors")
    parser.add_argument("--output-dir", type=str, default="prompt_based_feature_adaption/results",
                       help="Output directory for results")

    args = parser.parse_args()

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available()
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Run evaluation
    results = compare_with_baseline(
        category=args.category,
        checkpoint_path=Path(args.checkpoint),
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        k_neighbors=args.k_neighbors,
        device=device,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{args.category}_comparison.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {results_path}")


if __name__ == "__main__":
    main()
