"""
True Zero-shot Anomaly Detection using DINOv3.

This script implements a true zero-shot anomaly detection method where anomalies
are detected by comparing patch features within a single image, without needing
a reference set of normal images. It uses torch.hub to load DINOv3 models.
"""

import argparse
import torch
import cv2
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import os

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
def extract_embeddings(model, dataloader):
    """Extract embeddings using DINOv3."""
    patch_embeddings = []
    labels = []
    image_paths = []

    device = next(model.parameters()).device

    print("Extracting embeddings...")
    for batch in tqdm(dataloader):
        images = batch['image'].to(device)
        batch_labels = batch['label']
        batch_image_paths = batch['image_path']

        # Forward pass
        output = model(images, is_training=True)

        # DINOv3 returns dict
        patch_token = output['x_norm_patchtokens']

        patch_embeddings.append(patch_token.cpu().numpy())
        labels.append(batch_labels.numpy())
        image_paths.extend(batch_image_paths)

    result = {
        'patch_embeddings': np.concatenate(patch_embeddings, axis=0),
        'labels': np.concatenate(labels, axis=0),
        'image_paths': image_paths,
    }

    return result

def compute_anomaly_scores_zero_shot(patch_embeddings, k=1):
    """
    Compute anomaly scores for each image based on its own patch embeddings.
    patch_embeddings: (N, P, D) array, where N is number of images.
    """
    N, P, D = patch_embeddings.shape
    anomaly_scores = np.zeros(N)
    anomaly_maps = np.zeros((N, P))

    for i in tqdm(range(N), desc="Computing zero-shot anomaly scores"):
        # For each image, get its patch embeddings
        patches = patch_embeddings[i]  # (P, D)

        # Normalize
        patches_norm = patches / (np.linalg.norm(patches, axis=-1, keepdims=True) + 1e-8)

        # Compute pairwise cosine similarities between patches of the same image
        similarities = patches_norm @ patches_norm.T  # (P, P)

        # To exclude self-similarity, set diagonal to a very low value
        np.fill_diagonal(similarities, -1)

        if k == 1:
            # Max similarity for each patch to another patch in the same image
            patch_max_sim = similarities.max(axis=1)
        else:
            top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
            patch_max_sim = np.mean(
                np.take_along_axis(similarities, top_k_indices, axis=1),
                axis=1
            )

        # Anomaly score for each patch is 1 - max_similarity
        patch_anomaly_scores = 1 - patch_max_sim
        anomaly_maps[i] = patch_anomaly_scores

        # Image-level anomaly score is the maximum patch anomaly score
        anomaly_scores[i] = patch_anomaly_scores.max()

    return anomaly_scores, anomaly_maps

def evaluate(anomaly_scores, labels):
    """Evaluate anomaly detection performance."""
    auroc = roc_auc_score(labels, anomaly_scores)
    ap = average_precision_score(labels, anomaly_scores)

    return {
        'auroc': float(auroc),
        'average_precision': float(ap),
    }

def main():
    parser = argparse.ArgumentParser(description="Zero-shot anomaly detection with DINOv3")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument(
        "--root_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "mvtec_ad"),
        help="Root directory of the MVTec AD dataset",
    )
    parser.add_argument("--model-name", type=str, default="dinov3_vits16",
                       choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k-neighbors", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="./results")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(f"Zero-Shot Anomaly Detection: {args.category}")
    print("=" * 70 + "\n")

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model = load_dinov3_model(args.model_name)
    model = model.to(device)

    # Setup transforms and datasets
    transform = get_mvtec_transforms(224)

    print("\nLoading test data...")
    #
    # Create test dataset
    #
    project_root = Path(__file__).parent.parent
    test_dataset = MVTecADDataset(
        root=project_root / args.root_dir,
        category=args.category,
        split="test",
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    print(f"  Test samples: {len(test_dataset)}")
    print(f"    - Normal: {len(test_dataset.get_normal_samples())}")
    print(f"    - Anomaly: {len(test_dataset.get_anomaly_samples())}\n")

    # Extract embeddings
    test_embeddings = extract_embeddings(model, test_loader)

    # Use patch embeddings for zero-shot anomaly detection
    test_embed = test_embeddings['patch_embeddings']
    print("Using patch embeddings for zero-shot anomaly detection.")
    print(f"  Test shape: {test_embed.shape}")

    # Compute anomaly scores
    print(f"\nComputing zero-shot anomaly scores (k={args.k_neighbors})...")
    anomaly_scores, anomaly_maps = compute_anomaly_scores_zero_shot(test_embed, k=args.k_neighbors)

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(anomaly_scores, test_embeddings['labels'])

    print("\n" + "=" * 70)
    print(f"Results for {args.category}:")
    print("=" * 70)
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Average Precision: {metrics['average_precision']:.4f}")
    print("=" * 70 + "\n")

    # Generate and save high-resolution anomaly maps
    try:
        print("Generating anomaly maps...")
        base_output_dir = Path(args.output_dir) / args.category
        base_output_dir.mkdir(parents=True, exist_ok=True)

        map_size = int(np.sqrt(anomaly_maps.shape[1]))

        for i in tqdm(range(len(test_dataset)), desc="Generating anomaly maps"):
            img_path = test_embeddings['image_paths'][i]
            img_path_obj = Path(img_path)
            img = cv2.imread(str(img_path))

            # Get defect type and label for this image
            defect_type = test_dataset[i]['defect_type']
            label = test_dataset[i]['label']

            # Create output directory structure matching test structure
            output_dir_maps = base_output_dir / defect_type
            output_dir_maps.mkdir(parents=True, exist_ok=True)

            score_map = anomaly_maps[i]

            # Reshape score map to image size
            score_map_resized = cv2.resize(score_map.reshape(map_size, map_size), (img.shape[1], img.shape[0]))

            # Normalize for visualization
            norm_score_map = (score_map_resized - score_map_resized.min()) / (score_map_resized.max() - score_map_resized.min() + 1e-8)

            heatmap = cv2.applyColorMap((norm_score_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

            # Save overlay with metadata
            overlay_path = output_dir_maps / f"anomaly_map_{img_path_obj.stem}.png"
            
            # Convert OpenCV image to PIL for metadata support
            from PIL import Image
            from PIL.PngImagePlugin import PngInfo
            import datetime
            
            overlay_pil = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            
            # Add metadata to PNG
            metadata = PngInfo()
            metadata.add_text("source_image", str(img_path))
            metadata.add_text("defect_type", defect_type)
            metadata.add_text("label", "normal" if label == 0 else "anomaly")
            metadata.add_text("processing_timestamp", datetime.datetime.now().isoformat())
            metadata.add_text("category", args.category)
            metadata.add_text("model_name", args.model_name)
            metadata.add_text("k_neighbors", str(args.k_neighbors))
            metadata.add_text("relative_path", str(img_path_obj.relative_to(Path(args.root_dir).parent)))
            
            overlay_pil.save(overlay_path, "PNG", pnginfo=metadata)
            
        print(f"Anomaly maps generated and saved with folder structure to {base_output_dir}")
    except Exception as e:
        print(f"Could not generate anomaly maps: {e}")

    # Save results
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Save scores
    output_path = Path(args.output_dir) / args.category / "results.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, anomaly_scores=anomaly_scores, labels=test_embeddings['labels'])
    print(f"Saved anomaly scores to: {output_path}")


if __name__ == "__main__":
    main()
