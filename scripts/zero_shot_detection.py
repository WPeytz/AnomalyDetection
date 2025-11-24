#!/usr/bin/env python3
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
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from PIL import Image
import seaborn as sns

from mvtec_dataset import MVTecADDataset, get_mvtec_transforms

# Import SAM if available
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class FlippedDataset:
    """
    A wrapper dataset that horizontally flips all images.
    """
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        sample = self.original_dataset[idx]
        # Flip the image horizontally
        image = sample['image']
        flipped_image = torch.flip(image, dims=[-1])  # Flip along width dimension
        
        # Create new sample with all keys from original
        flipped_sample = {
            'image': flipped_image,
            'label': sample['label'],
            'image_path': sample['image_path']
        }
        
        # Add mask if it exists
        if 'mask' in sample:
            if sample['mask'] is not None:
                mask = sample['mask']
                if isinstance(mask, torch.Tensor):
                    flipped_sample['mask'] = torch.flip(mask, dims=[-1])
                else:
                    # Handle PIL Image by converting to array, flipping, and converting back
                    mask_array = np.array(mask)
                    flipped_mask_array = np.fliplr(mask_array)
                    flipped_sample['mask'] = Image.fromarray(flipped_mask_array)
            else:
                flipped_sample['mask'] = None
        else:
            flipped_sample['mask'] = None
            
        # Add defect_type if it exists
        if 'defect_type' in sample:
            flipped_sample['defect_type'] = sample['defect_type']
        else:
            flipped_sample['defect_type'] = 'good'  # Default for normal images
        
        return flipped_sample


class MirroringAnomalyDetector:
    """
    Implements the Mirroring DINO strategy: comparing an image with its flipped version.
    """
    def __init__(self, extractor, image_size=224, patch_size=14):
        self.extractor = extractor
        self.image_size = image_size
        self.patch_size = patch_size
        # Grid size: For DINOv3 with patch_size=14, we get 14x14=196 spatial patches
        self.grid_h = 14  # Actual spatial grid size for DINOv3
        self.grid_w = 14
        self.baseline_mean = 0
        self.baseline_std = 1

    def _align_flipped_patches(self, flipped_embeddings):
        """
        Re-aligns embeddings from a flipped image to match the original spatial grid.
        Input: (N, P, D)
        """
        N, P, D = flipped_embeddings.shape
        expected_patches = self.grid_h * self.grid_w
        
        # Detect and handle register tokens (usually 4 in DINOv3)
        n_registers = 0
        if P > expected_patches:
            n_registers = P - expected_patches
            
        # Separate registers and spatial tokens
        registers = flipped_embeddings[:, :n_registers, :]
        spatial = flipped_embeddings[:, n_registers:, :]
        
        # Check if spatial patches match expected grid
        current_P = spatial.shape[1]
        
        if current_P == expected_patches:
            # Perfect match: 196 spatial patches for 14x14 grid
            h, w = self.grid_h, self.grid_w
            
            try:
                spatial_grid = spatial.reshape(N, h, w, D)
                
                # Flip back horizontally (axis 2 is Width)
                spatial_grid_flipped = torch.flip(torch.tensor(spatial_grid), dims=[2]).numpy()
                
                # Flatten back
                spatial_restored = spatial_grid_flipped.reshape(N, current_P, D)
                
                # Recombine with registers
                if n_registers > 0:
                    return np.concatenate([registers, spatial_restored], axis=1)
                return spatial_restored
                
            except Exception as e:
                print(f"Warning: Patch reshaping failed: {e}")
                print("Using fallback: returning original embeddings")
                return flipped_embeddings
        else:
            # Mismatch: try to auto-detect grid or use fallback
            print(f"Warning: Spatial patch count mismatch.")
            print(f"Expected: {expected_patches} ({self.grid_h}x{self.grid_w}), Got: {current_P}")
            
            # Try to detect if it's a perfect square
            h = int(np.sqrt(current_P))
            if h * h == current_P:
                print(f"Auto-detected {h}x{h} grid for {current_P} patches")
                try:
                    spatial_grid = spatial.reshape(N, h, h, D)
                    spatial_grid_flipped = torch.flip(torch.tensor(spatial_grid), dims=[2]).numpy()
                    spatial_restored = spatial_grid_flipped.reshape(N, current_P, D)
                    
                    if n_registers > 0:
                        return np.concatenate([registers, spatial_restored], axis=1)
                    return spatial_restored
                    
                except Exception as e:
                    print(f"Auto-detection failed: {e}")
                    
            print("Using fallback: returning original embeddings")
            return flipped_embeddings

    def fit(self, train_loader):
        """
        'Training' phase: Compute symmetry error on normal data to establish a baseline.
        """
        print("Training Mirroring Model (Estimating Normal Symmetry Error)...")
        
        # For zero-shot, we don't have training data, so use test data to estimate baseline
        # This is a simplified version - in practice you'd use separate normal data
        return
        
    def predict(self, test_loader):
        """
        Evaluation phase: Compute symmetry error for test images.
        """
        # 1. Get Original Embeddings
        orig_results = self.extractor.extract_embeddings_batch(
            test_loader, extract_patches=True, extract_cls=False
        )
        orig_patches = orig_results['patch_embeddings']
        
        # 2. Get Flipped Embeddings
        test_dataset_flipped = FlippedDataset(test_loader.dataset)
        flipped_loader = DataLoader(
            test_dataset_flipped, 
            batch_size=test_loader.batch_size, 
            shuffle=False, 
            collate_fn=test_loader.collate_fn
        )
        
        flip_results = self.extractor.extract_embeddings_batch(
            flipped_loader, extract_patches=True, extract_cls=False
        )
        flip_patches = flip_results['patch_embeddings']
        
        # 3. Align Flipped Patches Back
        flip_patches_aligned = self._align_flipped_patches(flip_patches)
        
        # 4. Compute Patch-wise Cosine Distance
        # Normalize
        orig_norm = orig_patches / (np.linalg.norm(orig_patches, axis=-1, keepdims=True) + 1e-8)
        flip_norm = flip_patches_aligned / (np.linalg.norm(flip_patches_aligned, axis=-1, keepdims=True) + 1e-8)
        
        # Cosine similarity per patch
        sim_per_patch = (orig_norm * flip_norm).sum(axis=-1)
        dist_per_patch = 1 - sim_per_patch
        
        # Image-level score: Max patch error (most asymmetric part)
        image_scores = dist_per_patch.max(axis=1)
        
        return image_scores, dist_per_patch


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


class SimpleEmbeddingExtractor:
    """Simple wrapper for DINOv3 embedding extraction to work with MirroringAnomalyDetector."""
    def __init__(self, model):
        self.model = model
    
    def extract_embeddings_batch(self, dataloader, extract_patches=True, extract_cls=False):
        """Extract embeddings from a dataloader."""
        patch_embeddings = []
        labels = []
        image_paths = []
        
        device = next(self.model.parameters()).device
        
        print("Extracting embeddings...")
        with torch.no_grad():  # Add no_grad context
            for batch in tqdm(dataloader):
                images = batch['image'].to(device)
                batch_labels = batch['label']
                batch_paths = batch['image_path']
                
                # Extract features
                features = self.model.forward_features(images)
                patch_tokens = features['x_norm_patchtokens']
                
                patch_embeddings.append(patch_tokens.detach().cpu().numpy())  # Use detach()
                labels.extend(batch_labels.numpy())
                image_paths.extend(batch_paths)
        
        return {
            'patch_embeddings': np.concatenate(patch_embeddings, axis=0),
            'labels': np.array(labels),
            'image_paths': image_paths
        }


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

def create_anomaly_heatmap(patch_scores: np.ndarray, patch_size: int = 16, image_size: int = 224) -> np.ndarray:
    """
    Create spatial anomaly heatmap from patch-level scores.
    
    Args:
        patch_scores: Patch-level anomaly scores (N_patches,)
        patch_size: Size of each patch
        image_size: Original image size
        
    Returns:
        2D anomaly heatmap
    """
    # DINOv3 outputs register tokens at the beginning (typically 4 tokens)
    # For 224x224 image with 16x16 patches, we should have 14x14 = 196 spatial patches
    # But total patches may be 200 (4 register tokens + 196 spatial patches)
    
    n_patches = len(patch_scores)
    expected_grid_size = image_size // patch_size  # 14 for 224/16
    expected_patches = expected_grid_size * expected_grid_size  # 196
    
    # Skip register tokens if we have more patches than expected
    if n_patches > expected_patches:
        # Assume first (n_patches - expected_patches) are register tokens
        n_register_tokens = n_patches - expected_patches
        patch_scores = patch_scores[n_register_tokens:n_register_tokens + expected_patches]
    
    # Now reshape to grid
    if len(patch_scores) == expected_patches:
        heatmap = patch_scores.reshape(expected_grid_size, expected_grid_size)
    else:
        # Fallback: pad with zeros if needed
        padded_scores = np.zeros(expected_patches)
        padded_scores[:len(patch_scores)] = patch_scores
        heatmap = padded_scores.reshape(expected_grid_size, expected_grid_size)
    
    # Upsample to original image size using cubic interpolation for smoother results
    heatmap_resized = cv2.resize(heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    return heatmap_resized


def visualize_sam_enhanced_results(
    image: torch.Tensor,
    anomaly_score: float,
    anomaly_heatmap: Optional[np.ndarray],
    sam_mask: Optional[np.ndarray],
    label: int,
    defect_type: str,
    save_path: Path = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
):
    """
    Visualize SAM-enhanced anomaly detection results.
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Determine number of subplots
    if sam_mask is not None:
        n_plots = 4
    elif anomaly_heatmap is not None:
        n_plots = 3
    else:
        n_plots = 1
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Plot original image
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    # Plot anomaly heatmap if available
    if anomaly_heatmap is not None:
        import matplotlib.patches as patches
        
        im1 = axes[1].imshow(anomaly_heatmap, cmap='hot', alpha=0.7)
        axes[1].imshow(image_np, alpha=0.3)
        # Add bounding box to heatmap
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='lime', facecolor='none')
            axes[1].add_patch(rect)
        axes[1].set_title('Anomaly Heatmap', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot SAM mask if available
        if sam_mask is not None:
            axes[2].imshow(image_np)
            axes[2].imshow(sam_mask, alpha=0.5, cmap='Reds')
            axes[2].set_title('SAM Segmentation', fontsize=12)
            axes[2].axis('off')
            
            # Plot combined result
            combined_mask = anomaly_heatmap * sam_mask if sam_mask is not None else anomaly_heatmap
            axes[3].imshow(image_np)
            axes[3].imshow(combined_mask, alpha=0.7, cmap='hot')
            axes[3].set_title('SAM-Enhanced Result', fontsize=12)
            axes[3].axis('off')
        else:
            # Plot heatmap with bounding box (no SAM)
            axes[2].imshow(image_np)
            axes[2].imshow(anomaly_heatmap, alpha=0.7, cmap='hot')
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                        linewidth=2, edgecolor='lime', facecolor='none')
                axes[2].add_patch(rect)
            axes[2].set_title('Zero-Shot Anomaly Localization', fontsize=12)
            axes[2].axis('off')

    # Add overall title
    color = 'red' if label == 1 else 'green'
    title = f"Defect: {defect_type} | Anomaly Score: {anomaly_score:.3f}"
    fig.suptitle(title, color=color, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Save individual heatmap components separately
        if anomaly_heatmap is not None:
            save_dir = save_path.parent
            base_name = save_path.stem
            
            # Save heatmap only
            fig_heatmap, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(image_np)
            im = ax.imshow(anomaly_heatmap, alpha=0.7, cmap='hot')
            ax.set_title('Anomaly Heatmap (Zero-Shot)', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Extract image number from base_name
            if '_' in base_name:
                img_num = base_name.split('_')[0]
            else:
                img_num = base_name
            heatmap_path = save_dir / f"{img_num}_heatmap_only.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close(fig_heatmap)

    plt.close()


def evaluate_anomaly_detection(anomaly_scores, labels, threshold=None):
    """Evaluate anomaly detection performance with comprehensive metrics."""
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve
    
    # Compute AUC metrics
    auroc = roc_auc_score(labels, anomaly_scores)
    ap = average_precision_score(labels, anomaly_scores)
    
    # Determine threshold if not provided
    if threshold is None:
        # Use Youden's index (optimal threshold from ROC curve)
        fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]
    
    # Make binary predictions
    predictions = (anomaly_scores >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Extract TP, FP, TN, FN
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge case where only one class is present
        if len(np.unique(labels)) == 1:
            if labels[0] == 0:  # Only normal samples
                tn = len(labels) - np.sum(predictions)
                fp = np.sum(predictions)
                fn, tp = 0, 0
            else:  # Only anomaly samples
                tp = np.sum(predictions)
                fn = len(labels) - np.sum(predictions)
                tn, fp = 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
    
    # Compute additional metrics
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    # Compute accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Compute specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'auroc': float(auroc),
        'average_precision': float(ap),
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'confusion_matrix': cm.tolist()
    }


def plot_confusion_matrix(
    confusion_matrix,
    save_path,
    class_names=None,
    title="Confusion Matrix",
    normalize=False
):
    """
    Create and save a confusion matrix visualization.
    """
    if class_names is None:
        class_names = ['Normal', 'Anomaly']
    
    # Convert to numpy array if needed
    if isinstance(confusion_matrix, list):
        cm = np.array(confusion_matrix)
    else:
        cm = confusion_matrix
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, 
                annot=True, 
                fmt=fmt, 
                cmap='Blues',
                square=True,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add text annotations for better understanding
    if not normalize:
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Add labels
        plt.text(0.1, 0.1, f'TN\n{tn}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='darkblue')
        plt.text(1.1, 0.1, f'FP\n{fp}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='darkred')
        plt.text(0.1, 1.1, f'FN\n{fn}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='darkred')
        plt.text(1.1, 1.1, f'TP\n{tp}', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='darkgreen')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

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
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Enable saving visualizations (heatmaps and top samples)"
    )
    parser.add_argument(
        "--visualize-samples",
        type=int,
        default=5,
        help="Number of top anomaly/normal samples to visualize"
    )
    parser.add_argument(
        "--use-mirroring",
        action="store_true",
        help="Use mirroring strategy for anomaly detection"
    )

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

    # Check if mirroring is requested
    if args.use_mirroring:
        print("\nUsing Mirroring DINO strategy...")
        
        # Create embedding extractor wrapper
        extractor = SimpleEmbeddingExtractor(model)
        
        # Create mirroring detector
        mirroring_detector = MirroringAnomalyDetector(extractor, image_size=224, patch_size=16)
        
        # Fit (for zero-shot, this is mostly a no-op)
        mirroring_detector.fit(test_loader)
        
        # Get mirroring anomaly scores
        print("Evaluating with Mirroring Strategy...")
        anomaly_scores, patch_scores = mirroring_detector.predict(test_loader)
        
        # Create dummy anomaly maps for compatibility
        anomaly_maps = patch_scores.reshape(len(anomaly_scores), -1)
        
        method_name = "Mirroring"
    else:
        # Compute anomaly scores
        print(f"\nComputing zero-shot anomaly scores (k={args.k_neighbors})...")
        anomaly_scores, anomaly_maps = compute_anomaly_scores_zero_shot(test_embed, k=args.k_neighbors)
        method_name = f"Zero-Shot (k={args.k_neighbors})"

    # Evaluate with new function
    print("\nEvaluating...")
    metrics_cosine = evaluate_anomaly_detection(anomaly_scores, test_embeddings['labels'])
    
    # Create comprehensive metrics dict like all_models_script
    metrics = {
        'cosine': {**metrics_cosine, 'k': args.k_neighbors},
        'euclidean': {**metrics_cosine, 'k': args.k_neighbors},  # Same for zero-shot
        'knn': {**metrics_cosine, 'k': args.k_neighbors},  # Same for zero-shot
        'primary_metric': 'cosine',
    }

    print("\n" + "=" * 70)
    print(f"Results for {args.category}:")
    print("=" * 70)
    print(f"\n  {method_name}:")
    print(f"    AUROC: {metrics_cosine['auroc']:.4f}")
    print(f"    Average Precision: {metrics_cosine['average_precision']:.4f}")
    print("=" * 70 + "\n")

    # Save comprehensive metrics
    output_dir = Path(args.output_dir) / args.category
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Create and save confusion matrix visualizations
    if args.save_visualizations:
        cm_dir = output_dir / "confusion_matrices"
        cm_dir.mkdir(exist_ok=True)
        
        for metric_name, metric_data in metrics.items():
            if metric_name != 'primary_metric' and 'confusion_matrix' in metric_data:
                cm = metric_data['confusion_matrix']
                
                # Regular confusion matrix
                cm_path = cm_dir / f"confusion_matrix_{metric_name}.png"
                plot_confusion_matrix(
                    cm, 
                    cm_path, 
                    title=f"Confusion Matrix - {metric_name.title()} Metric"
                )
                
                # Normalized confusion matrix
                cm_norm_path = cm_dir / f"confusion_matrix_{metric_name}_normalized.png"
                plot_confusion_matrix(
                    cm, 
                    cm_norm_path, 
                    title=f"Confusion Matrix - {metric_name.title()} Metric",
                    normalize=True
                )

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

    # Generate comprehensive visualizations like all_models_script
    if args.save_visualizations:
        print(f"\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate anomaly heatmaps for all images
        anomaly_heatmaps = []
        for i in range(len(anomaly_scores)):
            heatmap = create_anomaly_heatmap(anomaly_maps[i], image_size=224)
            anomaly_heatmaps.append(heatmap)
        
        # Visualize top anomalies
        n_viz = args.visualize_samples
        anomaly_indices = np.argsort(anomaly_scores)[::-1][:n_viz]
        for i, idx in enumerate(anomaly_indices):
            sample = test_dataset[idx]
            image_path = Path(sample['image_path'])
            image_number = image_path.stem
            
            save_path = vis_dir / f"{image_number}_top_anomaly_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            heatmap = anomaly_heatmaps[idx] if idx < len(anomaly_heatmaps) else None
            
            visualize_sam_enhanced_results(
                sample['image'],
                anomaly_scores[idx],
                heatmap,
                None,  # No SAM mask for zero-shot
                sample['label'],
                sample['defect_type'],
                save_path,
                None,  # No bbox for zero-shot
            )
        
        # Visualize top normal samples
        normal_indices = np.argsort(anomaly_scores)[:n_viz]
        for i, idx in enumerate(normal_indices):
            sample = test_dataset[idx]
            image_path = Path(sample['image_path'])
            image_number = image_path.stem
            
            save_path = vis_dir / f"{image_number}_top_normal_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            heatmap = anomaly_heatmaps[idx] if idx < len(anomaly_heatmaps) else None
            
            visualize_sam_enhanced_results(
                sample['image'],
                anomaly_scores[idx],
                heatmap,
                None,  # No SAM mask for zero-shot
                sample['label'],
                sample['defect_type'],
                save_path,
                None,  # No bbox for zero-shot
            )
        
        print(f"Visualizations saved to {vis_dir}")
    else:
        print("Skipping visualizations (--save-visualizations not enabled)")

    # Save comprehensive anomaly scores like all_models_script
    scores_file = output_dir / "anomaly_scores.npz"
    np.savez(
        scores_file,
        scores_cosine=anomaly_scores,
        scores_euclidean=anomaly_scores,  # Same for zero-shot
        scores_knn=anomaly_scores,  # Same for zero-shot
        labels=test_embeddings['labels'],
        k_cosine=args.k_neighbors,
        k_euclidean=args.k_neighbors,
        k_knn=args.k_neighbors,
    )
    print(f"Saved anomaly scores to {scores_file}")


if __name__ == "__main__":
    main()
