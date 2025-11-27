"""
SAM-Enhanced Zero-shot and Few-shot Anomaly Detection using DINOv3

This script integrates SAM (Segment Anything Model) with DINOv3 for enhanced
anomaly detection and localization on MVTec AD dataset.

SAM Integration Features:
1. Zero-shot: Converts anomaly heatmaps into clean segmentation masks
2. Few-shot: Masks normal objects to create better prototypes
3. Provides sharp anomaly boundaries and removes background noise
"""

import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import cv2
from scipy import ndimage
from typing import Tuple, List, Optional
import warnings

# Import local modules
from embedding_extractor import DINOv3EmbeddingExtractor, compute_anomaly_scores
from mvtec_dataset import MVTecADDataset, get_mvtec_transforms

# Check SAM availability
try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("segment_anything package not installed")


def compute_patch_anomaly_scores_zeroshot(
    test_embeddings: np.ndarray,
    metric: str = 'cosine',
    save_dir: Optional[Path] = None,
    image_paths: Optional[List[str]] = None,
    image_size: int = 224,
    images: Optional[List[torch.Tensor]] = None,
    labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute patch-level anomaly scores using self-similarity (zero-shot).
    Each image is analyzed independently - anomalous patches are those
    that don't match other patches in the SAME image.
    
    GPU-accelerated version using PyTorch.
    
    Args:
        test_embeddings: Test patch embeddings (N, P, D)
        metric: Similarity metric ('cosine' or 'euclidean')
        save_dir: Directory to save heatmaps (optional)
        image_paths: List of image paths for saving (optional)
        image_size: Size to resize heatmaps to
        images: List of image tensors for visualization (optional)
        labels: Ground truth labels (0=normal, 1=anomaly) (optional)
    
    Returns:
        Patch anomaly scores (N, P) - higher means more anomalous
    """
    if test_embeddings.ndim != 3:
        raise ValueError("test_embeddings must be 3D (N, P, D)")
    
    N, P, D = test_embeddings.shape
    
    # Convert to torch tensors and move to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_torch = torch.from_numpy(test_embeddings).float().to(device)
    
    all_patch_scores = []
    
    # Process each image independently
    for img_idx in range(N):
        # Get patches for this single image: (P, D)
        patches = test_torch[img_idx]  # (P, D)
        
        if metric == 'cosine':
            # Normalize patches
            patches_norm = patches / (torch.norm(patches, dim=-1, keepdim=True) + 1e-8)
            
            # Compute self-similarity matrix: (P, P)
            similarity_matrix = patches_norm @ patches_norm.T
            
            # For each patch, compute statistics over similarities to OTHER patches
            # Set diagonal to -inf to exclude self-similarity
            similarity_matrix.fill_diagonal_(-float('inf'))
            
            # Use MEAN similarity instead of MAX to be more sensitive to outliers
            # Anomalous patches will have lower average similarity to other patches
            mean_similarities = similarity_matrix.mean(dim=1)  # (P,)
            
            # Alternative: use percentile-based approach (e.g., median or 25th percentile)
            # This is more robust to outliers
            # sorted_sims, _ = torch.sort(similarity_matrix, dim=1, descending=True)
            # Use top 50% of similarities (excluding self)
            # k_top = max(1, P // 2)
            # mean_similarities = sorted_sims[:, :k_top].mean(dim=1)
            
            max_similarities = mean_similarities
            
        else:  # euclidean
            # Compute pairwise distances: (P, P)
            distances = torch.cdist(patches.unsqueeze(0), patches.unsqueeze(0), p=2).squeeze(0)
            
            # Set diagonal to inf to exclude self
            distances.fill_diagonal_(float('inf'))
            
            # Use MEAN distance instead of MIN to be more sensitive
            mean_distances = distances.mean(dim=1)  # (P,)
            max_similarities = -mean_distances
        
        # Convert to anomaly scores
        patch_scores = 1 - max_similarities  # (P,)
        all_patch_scores.append(patch_scores)
    
    # Stack all results: (N, P)
    patch_anomaly_scores = torch.stack(all_patch_scores, dim=0)
    
    # Replace any NaN or Inf values before converting to numpy
    patch_anomaly_scores = torch.nan_to_num(patch_anomaly_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Convert to numpy
    patch_anomaly_scores = patch_anomaly_scores.cpu().numpy()
    
    # Normalize scores per image to [0, 1] range for better contrast
    # This makes each image's anomalies more visible
    for img_idx in range(N):
        img_scores = patch_anomaly_scores[img_idx]
        min_score = img_scores.min()
        max_score = img_scores.max()
        if max_score > min_score + 1e-8:  # Avoid division by zero
            patch_anomaly_scores[img_idx] = (img_scores - min_score) / (max_score - min_score)
        else:
            # If all scores are the same, set to 0 (no anomaly)
            patch_anomaly_scores[img_idx] = 0.0

    
    return patch_anomaly_scores


def compute_patch_anomaly_scores(
    test_embeddings: np.ndarray,
    normal_embeddings: np.ndarray,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Compute patch-level anomaly scores for heatmap visualization (few-shot).
    Compares test patches against normal training patches.
    
    GPU-accelerated version using PyTorch.
    
    Args:
        test_embeddings: Test patch embeddings (N, P, D)
        normal_embeddings: Normal patch embeddings (M, P, D)
        metric: Similarity metric ('cosine' or 'euclidean')
    
    Returns:
        Patch anomaly scores (N, P) - higher means more anomalous
    """
    if test_embeddings.ndim != 3:
        raise ValueError("test_embeddings must be 3D (N, P, D)")
    
    N, P, D = test_embeddings.shape
    M = normal_embeddings.shape[0]
    
    # Convert to torch tensors and move to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_torch = torch.from_numpy(test_embeddings).float().to(device)
    normal_torch = torch.from_numpy(normal_embeddings).float().to(device)
    
    # Flatten normal patches: (M, P, D) -> (M*P, D)
    normal_flat = normal_torch.reshape(M * P, D)
    
    if metric == 'cosine':
        # Normalize
        test_norm = test_torch / (torch.norm(test_torch, dim=-1, keepdim=True) + 1e-8)
        normal_norm = normal_flat / (torch.norm(normal_flat, dim=-1, keepdim=True) + 1e-8)
        
        # Compute similarities: (N, P, D) @ (D, M*P) = (N, P, M*P)
        similarities = test_norm @ normal_norm.T  # (N, P, M*P)
        
        # Max similarity for each patch
        max_similarities = similarities.max(dim=2)[0]  # (N, P)
        
    else:  # euclidean
        # Reshape test: (N, P, D) -> (N*P, D)
        test_flat = test_torch.reshape(N * P, D)
        
        # Compute all distances: (N*P, M*P)
        distances = torch.cdist(test_flat, normal_flat, p=2)
        
        # Reshape back: (N, P, M*P)
        distances = distances.reshape(N, P, M * P)
        
        # Min distance for each patch (negated for similarity)
        max_similarities = -distances.min(dim=2)[0]  # (N, P)
    
    # Convert to anomaly scores
    patch_anomaly_scores = 1 - max_similarities
    
    return patch_anomaly_scores.cpu().numpy()


class SAMIntegrator:
    """
    Integrates SAM with anomaly detection for enhanced localization.
    """
    
    def __init__(self, sam_checkpoint: Optional[str] = None, model_type: str = "vit_h", device: str = "cuda"):
        """
        Initialize SAM integrator.
        
        Args:
            sam_checkpoint: Path to SAM checkpoint file
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            device: Device to use ('cuda' or 'cpu')
        """
        self.sam_available = SAM_AVAILABLE and sam_checkpoint is not None
        self.predictor = None
        self.device = device if torch.cuda.is_available() else "cpu"
        
        if self.sam_available:
            try:
                print(f"Loading SAM model from {sam_checkpoint}...")
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                print(f"✓ SAM model loaded successfully: {model_type} on {self.device}")
            except Exception as e:
                print(f"Warning: Failed to load SAM model: {e}")
                self.sam_available = False
        else:
            if not SAM_AVAILABLE:
                print("SAM library not available - using fallback methods")
            else:
                print("SAM checkpoint not provided - using fallback methods")
    
    def get_bbox_from_anomaly_map(self, anomaly_map: np.ndarray, threshold_percentile: float = 85) -> Tuple[int, int, int, int]:
        """
        Extract bounding box from anomaly heatmap.
        
        Args:
            anomaly_map: 2D anomaly heatmap
            threshold_percentile: Percentile for thresholding (higher = less area, more conservative)
                                 Default 85 focuses on top 15% hottest regions
            
        Returns:
            Bounding box (x_min, y_min, x_max, y_max)
        """
        # Threshold the map - use higher percentile to focus on hottest regions only
        threshold = np.percentile(anomaly_map, threshold_percentile)
        binary_map = anomaly_map > threshold
        
        if not binary_map.any():
            # Fallback to center region if no anomalies found
            h, w = anomaly_map.shape
            return (w//4, h//4, 3*w//4, 3*h//4)
        
        # Find connected components and get largest one
        labeled, num_features = ndimage.label(binary_map)
        if num_features == 0:
            h, w = anomaly_map.shape
            return (w//4, h//4, 3*w//4, 3*h//4)
        
        # Get largest component
        component_sizes = ndimage.sum(binary_map, labeled, range(1, num_features + 1))
        largest_component = np.argmax(component_sizes) + 1
        largest_mask = labeled == largest_component
        
        # Get bounding box
        coords = np.where(largest_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        return (x_min, y_min, x_max, y_max)
    
    def predict_mask_from_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int], anomaly_map: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Predict segmentation mask using SAM from bounding box and anomaly heatmap.
        
        Args:
            image: RGB image (H, W, 3)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            anomaly_map: 2D anomaly heatmap to extract point prompts from hottest regions
            
        Returns:
            Binary mask or None if SAM not available
        """
        if not self.sam_available:
            return None
            
        try:
            self.predictor.set_image(image)
            
            # Use point prompts from heatmap peaks if available (NO BBOX for more precision)
            # Use the bounding box as SAM prompt (box) for segmentation
            if anomaly_map is not None:
                # Use the provided bbox to constrain SAM prediction
                x_min, y_min, x_max, y_max = bbox
                box_np = np.array([x_min, y_min, x_max, y_max])
                masks, scores, logits = self.predictor.predict(
                    box=box_np,
                    multimask_output=True,
                )
                # Select the mask with the highest score to prefer confident masks
                try:
                    best_idx = int(np.argmax(scores))
                except Exception:
                    best_idx = 0
                mask = masks[best_idx]
                # Apply morphological cleaning (conservative)
                mask = self._clean_mask(mask, kernel_size=5, min_area=100)
                return mask
            
            # Fallback: still use bbox if anomaly_map isn't available
            x_min, y_min, x_max, y_max = bbox
            box_np = np.array([x_min, y_min, x_max, y_max])
            masks, scores, logits = self.predictor.predict(
                box=box_np,
                multimask_output=True,
            )
            try:
                best_idx = int(np.argmax(scores))
            except Exception:
                best_idx = 0
            mask = masks[best_idx]
            mask = self._clean_mask(mask, kernel_size=5, min_area=100)
            return mask
            
        except Exception as e:
            print(f"SAM prediction failed: {e}")
            return None
    
    def _clean_mask(self, mask: np.ndarray, kernel_size: int = 3, min_area: int = 50) -> np.ndarray:
        """
        Clean up mask by removing isolated pixels and small noise using morphological operations.
        
        Args:
            mask: Binary mask to clean
            kernel_size: Size of morphological kernel (3 or 5 recommended)
            min_area: Minimum area to keep (removes smaller connected components)
            
        Returns:
            Cleaned binary mask
        """
        # Convert to uint8 for OpenCV
        mask_uint8 = mask.astype(np.uint8)
        
        # Create morphological kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Apply morphological opening (erosion followed by dilation)
        # This removes small isolated points and thin protrusions
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Optionally apply closing to fill small holes
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Remove small connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
        
        # Create output mask
        final_mask = np.zeros_like(mask_cleaned)
        
        # Keep only components larger than min_area (skip background label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                final_mask[labels == i] = 1
        
        return final_mask.astype(bool)
    
    def segment_normal_objects(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Segment normal objects using SAM auto-mask mode.
        
        Args:
            images: List of RGB images
            
        Returns:
            List of binary masks for normal objects
        """
        masks = []
        if not self.sam_available:
            # Fallback: return full image masks
            for img in images:
                masks.append(np.ones((img.shape[0], img.shape[1]), dtype=bool))
            return masks
        
        for img in images:
            try:
                self.predictor.set_image(img)
                # Use center point as prompt for main object
                h, w = img.shape[:2]
                center_point = np.array([[w//2, h//2]])
                center_label = np.array([1])
                
                pred_masks, scores, logits = self.predictor.predict(
                    point_coords=center_point,
                    point_labels=center_label,
                    multimask_output=True,
                )
                
                # Select mask with highest score
                best_mask = pred_masks[np.argmax(scores)]
                masks.append(best_mask)
            except Exception as e:
                print(f"SAM segmentation failed: {e}")
                # Fallback mask
                masks.append(np.ones((img.shape[0], img.shape[1]), dtype=bool))
        
        return masks


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


def load_ground_truth_mask(image_path: str, root_dir: str, category: str) -> Optional[np.ndarray]:
    """
    Load ground truth mask for a test image from MVTec dataset.
    
    Args:
        image_path: Path to the test image
        root_dir: Root directory of MVTec dataset
        category: Category name (e.g., 'bottle', 'cable')
    
    Returns:
        Binary ground truth mask (H, W) or None if not available
    """
    # Parse the image path to get defect type and filename
    # Expected structure: .../category/test/<defect_type>/<filename>.png
    image_path = Path(image_path)
    
    # Get defect type from path
    defect_type = image_path.parent.name
    
    # If it's a 'good' sample, there's no ground truth mask
    if defect_type == 'good':
        return None
    
    # Build ground truth path
    # Structure: root_dir/category/ground_truth/<defect_type>/<filename>_mask.png
    filename_stem = image_path.stem  # filename without extension
    
    # Try different possible naming conventions
    possible_paths = [
        Path(root_dir) / category / 'ground_truth' / defect_type / f"{filename_stem}_mask.png",
        Path(root_dir) / category / 'ground_truth' / defect_type / f"{filename_stem}.png",
        Path(root_dir) / category / 'groundtruth' / defect_type / f"{filename_stem}_mask.png",
        Path(root_dir) / category / 'groundtruth' / defect_type / f"{filename_stem}.png",
    ]
    
    for gt_path in possible_paths:
        if gt_path.exists():
            # Load and convert to binary mask
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if gt_mask is not None:
                # Convert to binary (threshold at 127)
                gt_mask = (gt_mask > 127).astype(np.uint8)
                return gt_mask
    
    return None


def evaluate_mask_segmentation(
    pred_masks: List[Optional[np.ndarray]],
    gt_masks: List[Optional[np.ndarray]],
) -> dict:
    """
    Evaluate segmentation mask quality against ground truth.
    Only evaluates anomalous samples (skips None ground truth masks).
    
    Args:
        pred_masks: List of predicted binary masks
        gt_masks: List of ground truth binary masks
    
    Returns:
        Dictionary with pixel-level evaluation metrics (Precision, Recall, Dice, Accuracy, IoU)
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    all_pred = []
    all_gt = []
    num_anomalous_samples = 0
    
    # Flatten all masks for pixel-level evaluation
    # Only process anomalous samples (where gt_mask is not None)
    for pred_mask, gt_mask in zip(pred_masks, gt_masks):
        # Skip if ground truth is None (normal/good samples)
        if gt_mask is None:
            continue
        
        # Skip if predicted mask is None
        if pred_mask is None:
            continue
        
        # Count anomalous samples
        num_anomalous_samples += 1
        
        # Ensure both masks are the same size
        if pred_mask.shape != gt_mask.shape:
            # Resize predicted mask to match ground truth
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                 (gt_mask.shape[1], gt_mask.shape[0]), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Flatten and collect
        all_pred.extend(pred_mask.flatten())
        all_gt.extend(gt_mask.flatten())
    
    if len(all_pred) == 0:
        return {
            'pixel_precision': 0.0,
            'pixel_recall': 0.0,
            'dice_coefficient': 0.0,
            'pixel_accuracy': 0.0,
            'iou': 0.0,
            'num_evaluated_pixels': 0,
            'num_anomalous_samples': 0,
        }
    
    # Convert to numpy arrays
    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)
    
    # Compute basic metrics
    accuracy = accuracy_score(all_gt, all_pred)
    
    # Precision and recall (handle edge cases)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        precision = precision_score(all_gt, all_pred, zero_division=0)
        recall = recall_score(all_gt, all_pred, zero_division=0)
    
    # Compute Dice coefficient (F1 score for binary segmentation)
    # Dice = 2 * TP / (2 * TP + FP + FN) = 2 * Precision * Recall / (Precision + Recall)
    if precision + recall > 0:
        dice = 2 * precision * recall / (precision + recall)
    else:
        dice = 0.0
    
    # Compute IoU (Intersection over Union)
    # IoU = TP / (TP + FP + FN)
    intersection = np.sum(all_pred & all_gt)
    union = np.sum(all_pred | all_gt)
    if union > 0:
        iou = intersection / union
    else:
        iou = 0.0
    
    return {
        'pixel_precision': float(precision),
        'pixel_recall': float(recall),
        'dice_coefficient': float(dice),
        'pixel_accuracy': float(accuracy),
        'iou': float(iou),
        'num_evaluated_pixels': len(all_pred),
        'num_anomalous_samples': num_anomalous_samples,
    }


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
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Compute metrics
    auroc = roc_auc_score(labels, anomaly_scores)
    ap = average_precision_score(labels, anomaly_scores)
    
    # Compute precision and recall at optimal threshold (maximize F1)
    precision_vals, recall_vals, thresholds = precision_recall_curve(labels, anomaly_scores)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_precision = precision_vals[optimal_idx]
    optimal_recall = recall_vals[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]

    return {
        'auroc': float(auroc),
        'average_precision': float(ap),
        'precision': float(optimal_precision),
        'recall': float(optimal_recall),
        'f1_score': float(optimal_f1),
    }


def visualize_sam_enhanced_results(
    image: torch.Tensor,
    anomaly_score: float,
    anomaly_heatmap: Optional[np.ndarray],
    sam_mask: Optional[np.ndarray],
    label: int,
    defect_type: str,
    save_path: Path = Path(__file__).parent,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    gt_mask: Optional[np.ndarray] = None,
):
    """
    Visualize SAM-enhanced anomaly detection results.

    Args:
        image: Image tensor (C, H, W)
        anomaly_score: Overall anomaly score
        anomaly_heatmap: Spatial anomaly heatmap
        sam_mask: SAM segmentation mask
        label: Ground truth label
        defect_type: Defect type name
        save_path: Path to save the visualization
        bbox: Bounding box from anomaly heatmap (x_min, y_min, x_max, y_max)
        gt_mask: Ground truth mask for comparison (optional)
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Determine number of subplots based on what's available
    # If gt_mask is provided, show: original, gt_mask, predicted_mask (3 plots)
    if gt_mask is not None and sam_mask is not None:
        n_plots = 3
    # If no SAM mask, we need 3 plots: original image, heatmap, heatmap with bbox
    # If SAM mask is available, we need 4 plots: original, heatmap, SAM mask, combined
    elif sam_mask is not None:
        n_plots = 4
    elif anomaly_heatmap is not None:
        n_plots = 3
    else:
        n_plots = 1
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Special case: ground truth comparison mode (3 plots: image, heatmap, gt+pred overlay)
    # Only show this for anomalous samples (when gt_mask is not None)
    if gt_mask is not None and sam_mask is not None:
        # Plot 1: Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Plot 2: Anomaly heatmap
        if anomaly_heatmap is not None:
            axes[1].imshow(image_np)
            im = axes[1].imshow(anomaly_heatmap, alpha=0.7, cmap='hot')
            axes[1].set_title('Anomaly Heatmap', fontsize=12)
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        else:
            axes[1].imshow(image_np)
            axes[1].set_title('No Heatmap', fontsize=12)
            axes[1].axis('off')
        
        # Plot 3: Ground truth (green) + Predicted mask (red) overlay
        axes[2].imshow(image_np)
        # Resize masks to match image size (important: gt masks come from original files)
        img_h, img_w = image_np.shape[:2]
        # Ensure gt_mask is numpy
        if not isinstance(gt_mask, np.ndarray):
            gt_mask_arr = np.array(gt_mask)
        else:
            gt_mask_arr = gt_mask
        if not isinstance(sam_mask, np.ndarray):
            sam_mask_arr = np.array(sam_mask)
        else:
            sam_mask_arr = sam_mask

        # Resize with nearest neighbor to preserve binary labels
        try:
            if gt_mask_arr.shape[:2] != (img_h, img_w):
                gt_resized = cv2.resize(gt_mask_arr.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                gt_resized = gt_mask_arr.astype(np.uint8)
        except Exception:
            gt_resized = (gt_mask_arr > 0).astype(np.uint8)

        try:
            if sam_mask_arr.shape[:2] != (img_h, img_w):
                sam_resized = cv2.resize(sam_mask_arr.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                sam_resized = sam_mask_arr.astype(np.uint8)
        except Exception:
            sam_resized = (sam_mask_arr > 0).astype(np.uint8)

        # Convert masks to float and ensure they're binary
        gt_normalized = (gt_resized > 0).astype(np.float32)
        sam_normalized = (sam_resized > 0).astype(np.float32)

        # Show ground truth in green with higher alpha for visibility
        axes[2].imshow(np.stack([np.zeros_like(gt_normalized), gt_normalized, np.zeros_like(gt_normalized)], axis=-1),
                      alpha=0.5)
        # Show predicted mask in red with higher alpha
        axes[2].imshow(np.stack([sam_normalized, np.zeros_like(sam_normalized), np.zeros_like(sam_normalized)], axis=-1),
                      alpha=0.5)
        axes[2].set_title('GT (Green) + Pred (Red) Overlay', fontsize=12)
        axes[2].axis('off')
    else:
        # Original visualization logic
        # Plot original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')

        # Plot anomaly heatmap if available (only in non-gt-comparison mode)
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
                # Add bounding box to SAM mask
                axes[2].set_title('SAM Segmentation', fontsize=12)
                axes[2].axis('off')
                
                # Plot combined result
                combined_mask = anomaly_heatmap * sam_mask if sam_mask is not None else anomaly_heatmap
                axes[3].imshow(image_np)
                axes[3].imshow(combined_mask, alpha=0.7, cmap='hot')
                # Add bounding box to combined result
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
                axes[2].set_title('DINOv3 Anomaly Localization', fontsize=12)
                axes[2].axis('off')

    # Add overall title
    color = 'red' if label == 1 else 'green'
    title = f"Defect: {defect_type} | Anomaly Score: {anomaly_score:.3f}"
    fig.suptitle(title, color=color, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved SAM visualization to {save_path}")
        
        # Save individual heatmap components separately
        if anomaly_heatmap is not None:
            save_dir = save_path.parent
            base_name = save_path.stem
            
            # Save heatmap only
            fig_heatmap, ax = plt.subplots(1, 1, figsize=(5, 5))
            ax.imshow(image_np)
            im = ax.imshow(anomaly_heatmap, alpha=0.7, cmap='hot')
            ax.set_title('Anomaly Heatmap (No SAM)', fontsize=12)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            # Extract image number from base_name if present, otherwise use base_name as is
            if '_' in base_name:
                img_num = base_name.split('_')[0]
            else:
                img_num = base_name
            heatmap_path = save_dir / f"{img_num}_heatmap_only.png"
            plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
            plt.close(fig_heatmap)
            
            # Save SAM-enhanced heatmap if available
            if sam_mask is not None:
                fig_sam, ax = plt.subplots(1, 1, figsize=(5, 5))
                combined_mask = anomaly_heatmap * sam_mask
                ax.imshow(image_np)
                im = ax.imshow(combined_mask, alpha=0.7, cmap='hot')
                ax.set_title('SAM-Enhanced Heatmap', fontsize=12)
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Extract image number from base_name if present, otherwise use base_name as is
                if '_' in base_name:
                    img_num = base_name.split('_')[0]
                else:
                    img_num = base_name
                sam_path = save_dir / f"{img_num}_heatmap_with_sam.png"
                plt.savefig(sam_path, dpi=150, bbox_inches='tight')
                plt.close(fig_sam)
    else:
        plt.show()

    plt.close()


def visualize_anomaly_map(
    image: torch.Tensor,
    anomaly_score: float,
    label: int,
    defect_type: str,
    save_path: Path = None,
):
    """
    Visualize an image with its anomaly score (legacy function).
    """
    visualize_sam_enhanced_results(
        image, anomaly_score, None, None, label, defect_type, save_path, None, None
    )


def run_sam_enhanced_anomaly_detection(
    category: str,
    root_dir: str = "mvtec_ad",
    model_name: str = "dinov3_vits16",
    batch_size: int = 32,
    image_size: int = 224,
    use_patches: bool = True,
    k_neighbors: int = 3,
    output_dir: str = "results",
    visualize_samples: int = 5,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_l",
    use_sam_for_normal_masking: bool = True,
    few_shot_mode: bool = False,
    n_shots: int = 5,
    save_visualizations: bool = True,
):
    """
    Run SAM-enhanced anomaly detection on MVTec AD category.

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
        sam_checkpoint: Path to SAM checkpoint file
        use_sam_for_normal_masking: Whether to use SAM for masking normal objects
        few_shot_mode: Whether to use few-shot learning
        n_shots: Number of shots for few-shot learning
    """
    mode_str = f"Few-Shot ({n_shots} shots)" if few_shot_mode else "Zero-Shot"
    print(f"\n{'='*70}")
    print(f"SAM-Enhanced {mode_str} Anomaly Detection: {category}")
    print(f"{'='*70}\n")

    # Setup output directory - make it relative to script location
    # Output dir is already configured by main() for multi-category runs
    script_dir = Path(__file__).parent
    output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_integrator = SAMIntegrator(sam_checkpoint, model_type=sam_model_type, device=device)

    # Initialize DINOv3 model
    print(f"Loading model: {model_name}")
    extractor = DINOv3EmbeddingExtractor(
        model_name=model_name,
        use_huggingface=True,
    )

    # Setup transforms
    transform = get_mvtec_transforms(image_size)

    # Load training data (only normal samples)
    print(f"\nLoading training data...")
    full_train_dataset = MVTecADDataset(
        root=root_dir,
        category=category,
        split='train',
        transform=transform,
    )
    
    # Apply few-shot sampling if needed
    if few_shot_mode and n_shots > 0:
        from torch.utils.data import Subset
        import random
        normal_indices = full_train_dataset.get_normal_samples()
        if len(normal_indices) > n_shots:
            random.seed(42)
            selected_indices = random.sample(normal_indices, n_shots)
            train_dataset = Subset(full_train_dataset, selected_indices)
        else:
            train_dataset = full_train_dataset
        print(f"Few-shot training samples: {len(train_dataset)}")
    else:
        # Zero-shot (n_shots=0) or regular mode: use all training samples
        train_dataset = full_train_dataset
        if few_shot_mode and n_shots == 0:
            print(f"Zero-shot mode: using all {len(train_dataset)} training samples")
        else:
            print(f"Training samples: {len(train_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # SAM-Enhanced Normal Prototype Creation
    if use_sam_for_normal_masking and sam_integrator.sam_available:
        print("Creating SAM-enhanced normal prototypes...")
        normal_images = []
        for i in range(min(len(train_dataset), 10)):  # Use first 10 samples for SAM masking
            sample = train_dataset[i] if hasattr(train_dataset, '__getitem__') else full_train_dataset[train_dataset.indices[i]]
            # Convert tensor to numpy for SAM
            img_tensor = sample['image']
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_denorm = img_tensor * std + mean
            img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            normal_images.append(img_np)
        
        # Get SAM masks for normal objects
        normal_masks = sam_integrator.segment_normal_objects(normal_images)
        print(f"Generated {len(normal_masks)} SAM masks for normal objects")

    # Extract training embeddings
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

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

    # Compute anomaly scores for all metrics
    print(f"\nComputing anomaly scores for all metrics...")
    
    # Check if zero-shot mode (no few-shot flag or n_shots not provided)
    is_zeroshot = not few_shot_mode
    
    # Determine k values for each metric
    k_cosine = k_neighbors
    k_euclidean = 1  # Use nearest neighbor for euclidean
    k_knn = k_neighbors  # Use k-NN with k neighbors
    
    if use_patches:
        if is_zeroshot:
            # TRUE ZERO-SHOT: Use self-similarity (no reference to training data)
            print(f"Using TRUE ZERO-SHOT mode (patch self-similarity)...")
            
            # For image-level scores, use mean of patch self-similarity scores
            print(f"Computing zero-shot patch self-similarity scores...")
            
            # Get image paths for saving heatmaps
            image_paths = [test_dataset[i]['image_path'] for i in range(len(test_dataset))]
            
            # Get images and labels for visualization
            test_images = [test_dataset[i]['image'] for i in range(len(test_dataset))]
            test_labels = test_embeddings['labels']
            
            # Compute zero-shot heatmaps and save them
            zeroshot_heatmap_dir = output_dir / "zeroshot_heatmaps"
            patch_level_scores = compute_patch_anomaly_scores_zeroshot(
                test_embed,
                metric='cosine',
                save_dir=zeroshot_heatmap_dir if save_visualizations else None,
                image_paths=image_paths,
                image_size=image_size,
                images=test_images if save_visualizations else None,
                labels=test_labels,
            )
            
            # Image-level anomaly score = mean of patch anomaly scores
            anomaly_scores_cosine = patch_level_scores.mean(axis=1)
            anomaly_scores_euclidean = anomaly_scores_cosine.copy()  # Same for zero-shot
            anomaly_scores_knn = anomaly_scores_cosine.copy()  # Same for zero-shot
            
            print(f"Zero-shot patch-level scores shape: {patch_level_scores.shape}")
            print(f"Zero-shot image-level scores shape: {anomaly_scores_cosine.shape}")
        else:
            # FEW-SHOT: Compare against training samples
            print(f"Using FEW-SHOT mode ({n_shots} shots)...")
            
            # Cosine similarity
            print(f"  Cosine similarity (k={k_cosine})...")
            anomaly_scores_cosine = compute_anomaly_scores(
                test_embed,
                normal_embeddings,
                metric='cosine',
                k=k_cosine,
            )
            
            # Euclidean distance
            print(f"  Euclidean distance (k={k_euclidean})...")
            anomaly_scores_euclidean = compute_anomaly_scores(
                test_embed,
                normal_embeddings,
                metric='euclidean',
                k=k_euclidean,
            )
            
            # k-NN distance
            print(f"  k-NN distance (k={k_knn})...")
            anomaly_scores_knn = compute_anomaly_scores(
                test_embed,
                normal_embeddings,
                metric='knn',
                k=k_knn,
            )
            
            # Compute patch-level scores for heatmap visualization (using cosine)
            if test_embed.ndim == 3:  # (N, P, D) format
                print(f"Computing patch-level scores for heatmaps...")
                patch_level_scores = compute_patch_anomaly_scores(
                    test_embed,
                    normal_embeddings,
                    metric='cosine',
                )
                print(f"Patch-level scores shape: {patch_level_scores.shape}")
            else:
                patch_level_scores = None
        
        # Use cosine as primary metric (backward compatibility)
        anomaly_scores = anomaly_scores_cosine
        print(f"Image-level anomaly scores shape: {anomaly_scores.shape}")
        
    else:
        # CLS token anomaly detection - always use few-shot approach
        if is_zeroshot:
            print("Warning: CLS token mode does not support true zero-shot. Using all training samples.")
        
        print(f"  Cosine similarity (k={k_cosine})...")
        anomaly_scores_cosine = compute_anomaly_scores(
            test_embed,
            normal_embeddings,
            metric='cosine',
            k=k_cosine,
        )
        
        print(f"  Euclidean distance (k={k_euclidean})...")
        anomaly_scores_euclidean = compute_anomaly_scores(
            test_embed,
            normal_embeddings,
            metric='euclidean',
            k=k_euclidean,
        )
        
        # k-NN distance
        print(f"  k-NN distance (k={k_knn})...")
        anomaly_scores_knn = compute_anomaly_scores(
            test_embed,
            normal_embeddings,
            metric='knn',
            k=k_knn,
        )
        
        # Use cosine as primary metric
        anomaly_scores = anomaly_scores_cosine
        patch_level_scores = None

    print(f"Computed anomaly scores for {len(anomaly_scores)} test images")

    # Generate anomaly heatmaps for all images (only if visualizations are needed)
    sam_masks = []
    anomaly_heatmaps = []
    bboxes = []
    
    if save_visualizations and patch_level_scores is not None:
        print(f"Generating anomaly heatmaps for {len(anomaly_scores)} images...")
        print(f"Patch scores per image: {patch_level_scores.shape[1]}")
        
        # Generate heatmaps and bounding boxes for all images
        for i in range(len(anomaly_scores)):
            heatmap = create_anomaly_heatmap(patch_level_scores[i], image_size=image_size)
            anomaly_heatmaps.append(heatmap)
            
            # Get bounding box from anomaly heatmap
            bbox = sam_integrator.get_bbox_from_anomaly_map(heatmap)
            bboxes.append(bbox)
        
        # Apply SAM post-processing if available
        if sam_integrator.sam_available:
            print("Applying SAM post-processing (anomalous samples only)...")
            
            for i in range(len(anomaly_scores)):
                # Get test image
                sample = test_dataset[i]
                
                # Only apply SAM to anomalous samples (label=1)
                # Skip normal samples (label=0 or defect_type='good')
                if sample['label'] == 0 or sample['defect_type'] == 'good':
                    sam_masks.append(None)
                    continue
                
                img_tensor = sample['image']
                
                # Convert to numpy for SAM
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = img_tensor * std + mean
                img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Generate SAM mask using the pre-computed bbox and heatmap
                sam_mask = sam_integrator.predict_mask_from_bbox(img_np, bboxes[i], anomaly_heatmaps[i])
                sam_masks.append(sam_mask)
                
            num_sam_masks = sum(1 for m in sam_masks if m is not None)
            print(f"Generated {num_sam_masks} SAM masks for anomalous samples")
        else:
            print("SAM not available - heatmaps generated with bounding boxes")
    elif not save_visualizations:
        print("Skipping heatmap generation (--save-visualizations not enabled)")
    
    # Evaluate SAM masks against ground truth if --use-sam-masking is enabled
    mask_metrics = None
    gt_masks = []
    if use_sam_for_normal_masking and sam_integrator.sam_available and len(sam_masks) > 0:
        print(f"\nLoading ground truth masks for mask evaluation...")
        
        # Load ground truth masks for all test images
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            gt_mask = load_ground_truth_mask(sample['image_path'], root_dir, category)
            gt_masks.append(gt_mask)
        
        print(f"Loaded {sum(1 for m in gt_masks if m is not None)} ground truth masks")
        
        # Evaluate SAM masks against ground truth
        print(f"Evaluating SAM mask quality (anomalous samples only)...")
        mask_metrics = evaluate_mask_segmentation(sam_masks, gt_masks)
        
        print(f"\nMask Segmentation Metrics (Anomalous Samples Only):")
        print(f"  Pixel Precision: {mask_metrics['pixel_precision']:.4f}")
        print(f"  Pixel Recall: {mask_metrics['pixel_recall']:.4f}")
        print(f"  Dice Coefficient: {mask_metrics['dice_coefficient']:.4f}")
        print(f"  Pixel Accuracy: {mask_metrics['pixel_accuracy']:.4f}")
        print(f"  IoU: {mask_metrics['iou']:.4f}")
        print(f"  Number of anomalous samples: {mask_metrics['num_anomalous_samples']}")
        print(f"  Number of pixels evaluated: {mask_metrics['num_evaluated_pixels']}")
        
        # Save mask metrics to file
        mask_metrics_file = output_dir / "mask_segmentation_metrics.json"
        with open(mask_metrics_file, 'w') as f:
            json.dump(mask_metrics, f, indent=2)
        print(f"Saved mask segmentation metrics to {mask_metrics_file}")
    
    # Evaluate all metrics
    print(f"\nEvaluating metrics...")
    metrics_cosine = evaluate_anomaly_detection(anomaly_scores_cosine, test_embeddings['labels'])
    metrics_euclidean = evaluate_anomaly_detection(anomaly_scores_euclidean, test_embeddings['labels'])
    metrics_knn = evaluate_anomaly_detection(anomaly_scores_knn, test_embeddings['labels'])
    
    # Primary metrics dict with all results
    metrics = {
        'cosine': {**metrics_cosine, 'k': k_cosine},
        'euclidean': {**metrics_euclidean, 'k': k_euclidean},
        'knn': {**metrics_knn, 'k': k_knn},
        'primary_metric': 'cosine',
    }

    print(f"\n{'='*70}")
    print(f"Results for {category}:")
    print(f"{'='*70}")
    print(f"\n  Cosine Similarity (k={k_cosine}):")
    print(f"    AUROC: {metrics_cosine['auroc']:.4f}")
    print(f"    Average Precision: {metrics_cosine['average_precision']:.4f}")
    print(f"    Precision: {metrics_cosine['precision']:.4f}")
    print(f"    Recall: {metrics_cosine['recall']:.4f}")
    print(f"    F1 Score: {metrics_cosine['f1_score']:.4f}")
    print(f"\n  Euclidean Distance (k={k_euclidean}):")
    print(f"    AUROC: {metrics_euclidean['auroc']:.4f}")
    print(f"    Average Precision: {metrics_euclidean['average_precision']:.4f}")
    print(f"    Precision: {metrics_euclidean['precision']:.4f}")
    print(f"    Recall: {metrics_euclidean['recall']:.4f}")
    print(f"    F1 Score: {metrics_euclidean['f1_score']:.4f}")
    print(f"\n  k-NN Distance (k={k_knn}):")
    print(f"    AUROC: {metrics_knn['auroc']:.4f}")
    print(f"    Average Precision: {metrics_knn['average_precision']:.4f}")
    print(f"    Precision: {metrics_knn['precision']:.4f}")
    print(f"    Recall: {metrics_knn['recall']:.4f}")
    print(f"    F1 Score: {metrics_knn['f1_score']:.4f}")
    print(f"{'='*70}\n")

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Visualize samples
    if save_visualizations and visualize_samples > 0:
        print(f"\nGenerating visualizations...")
        # Create visualization subdirectory within the results folder
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Visualize top anomalies
        anomaly_indices = np.argsort(anomaly_scores)[::-1][:visualize_samples]

        for i, idx in enumerate(anomaly_indices):
            sample = test_dataset[idx]
            
            # Extract image number from the image path (e.g., "xxx.png" -> "xxx")
            image_path = Path(sample['image_path'])
            image_number = image_path.stem  # Gets filename without extension
            
            save_path = vis_dir / f"{image_number}_top_anomaly_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            
            # Get heatmap, SAM mask, and bbox if available
            heatmap = anomaly_heatmaps[idx] if idx < len(anomaly_heatmaps) else None
            sam_mask_data = sam_masks[idx] if idx < len(sam_masks) else None
            bbox_data = bboxes[idx] if idx < len(bboxes) else None
            # Only pass ground truth for anomalous samples (when --use-sam-masking is on)
            gt_mask_data = gt_masks[idx] if (idx < len(gt_masks) and use_sam_for_normal_masking) else None
            
            # Save SAM mask as .npy file if available
            if sam_mask_data is not None:
                mask_save_path = vis_dir / f"{image_number}_sam_mask.npy"
                np.save(mask_save_path, sam_mask_data)
            
            visualize_sam_enhanced_results(
                sample['image'],
                anomaly_scores[idx],
                heatmap,
                sam_mask_data,
                sample['label'],
                sample['defect_type'],
                save_path,
                bbox_data,
                gt_mask_data,
            )

        # Visualize top normal samples
        normal_indices = np.argsort(anomaly_scores)[:visualize_samples]

        for i, idx in enumerate(normal_indices):
            sample = test_dataset[idx]
            
            # Extract image number from the image path (e.g., "xxx.png" -> "xxx")
            image_path = Path(sample['image_path'])
            image_number = image_path.stem  # Gets filename without extension
            
            save_path = vis_dir / f"{image_number}_top_normal_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            
            # Get heatmap, SAM mask, and bbox if available
            heatmap = anomaly_heatmaps[idx] if idx < len(anomaly_heatmaps) else None
            sam_mask_data = sam_masks[idx] if idx < len(sam_masks) else None
            bbox_data = bboxes[idx] if idx < len(bboxes) else None
            # Don't pass ground truth for normal samples - they don't have ground truth masks
            gt_mask_data = None
            
            # Save SAM mask as .npy file if available
            if sam_mask_data is not None:
                mask_save_path = vis_dir / f"{image_number}_sam_mask.npy"
                np.save(mask_save_path, sam_mask_data)
            
            visualize_sam_enhanced_results(
                sample['image'],
                anomaly_scores[idx],
                heatmap,
                sam_mask_data,
                sample['label'],
                sample['defect_type'],
                save_path,
                bbox_data,
                gt_mask_data,
            )

    # Save anomaly scores for all metrics
    scores_file = output_dir / "anomaly_scores.npz"
    np.savez(
        scores_file,
        scores_cosine=anomaly_scores_cosine,
        scores_euclidean=anomaly_scores_euclidean,
        scores_knn=anomaly_scores_knn,
        labels=test_embeddings['labels'],
        k_cosine=k_cosine,
        k_euclidean=k_euclidean,
        k_knn=k_knn,
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
        nargs='+',
        default=["bottle"],
        choices=[
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper', 'all'
        ],
        help="MVTec AD categories to evaluate (one or more categories, or 'all' for all categories). Example: --category bottle carpet zipper"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=str(Path(__file__).parent.parent / "mvtec_ad"),
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
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        default=False,
        help="Enable saving visualizations (heatmaps will only be computed if this is set)"
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default=None,
        help="Path to SAM checkpoint file (e.g., sam_vit_h_4b8939.pth)"
    )
    parser.add_argument(
        "--use-sam",
        action="store_true",
        help="Enable SAM for anomaly segmentation and visualization enhancement"
    )
    parser.add_argument(
        "--use-sam-masking",
        action="store_true",
        help="Use SAM for masking normal objects in prototypes (requires --use-sam)"
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="Enable few-shot learning mode"
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=5,
        help="Number of shots for few-shot learning"
    )
    parser.add_argument(
        "--shots-array",
        type=str,
        default=None,
        help="Array of shot values to test, e.g., '1,3,5,10' (overrides --n-shots)"
    )

    args = parser.parse_args()

    # Parse shots array if provided
    shots_to_test = None
    if args.shots_array:
        try:
            shots_to_test = [int(x.strip()) for x in args.shots_array.split(',')]
            print(f"Testing multiple shot values: {shots_to_test}")
        except ValueError:
            print(f"Error parsing --shots-array: {args.shots_array}. Expected format: '1,3,5,10'")
            return
    else:
        # Single shot value mode
        shots_to_test = [args.n_shots] if args.few_shot else None

    # Only use SAM if --use-sam flag is provided
    sam_model_type = "vit_h"  # default
    
    if args.use_sam:
        # Auto-detect SAM checkpoint if not provided
        if args.sam_checkpoint is None:
            script_dir = Path(__file__).parent
            # Check for SAM checkpoints in order of preference (largest/best first)
            sam_checkpoints = [
                script_dir / "sam_vit_h_4b8939.pth",
                script_dir / "sam_vit_l_0b3195.pth",
                script_dir / "sam_vit_b_01ec64.pth",
            ]
            for checkpoint in sam_checkpoints:
                if checkpoint.exists():
                    args.sam_checkpoint = str(checkpoint)
                    # Determine model type from filename
                    if "vit_h" in checkpoint.name:
                        sam_model_type = "vit_h"
                    elif "vit_l" in checkpoint.name:
                        sam_model_type = "vit_l"
                    elif "vit_b" in checkpoint.name:
                        sam_model_type = "vit_b"
                    print(f"Auto-detected SAM checkpoint: {checkpoint.name}")
                    break
            
            if args.sam_checkpoint is None:
                print("Warning: --use-sam specified but no SAM checkpoint found. Running without SAM.")
                args.use_sam = False
        else:
            # Determine model type from provided checkpoint filename
            checkpoint_name = Path(args.sam_checkpoint).name
            if "vit_h" in checkpoint_name:
                sam_model_type = "vit_h"
            elif "vit_l" in checkpoint_name:
                sam_model_type = "vit_l"
            elif "vit_b" in checkpoint_name:
                sam_model_type = "vit_b"
    
    # If not using SAM, ensure checkpoint is None
    if not args.use_sam:
        args.sam_checkpoint = None

    # Run on all categories or specified categories
    if 'all' in args.category:
        categories = [
            'bottle', 'cable', 'capsule', 'carpet', 'grid',
            'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
            'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
        ]
    else:
        categories = args.category
    
    # Create folder name from categories
    if 'all' in args.category:
        category_folder_name = 'all_categories'
    elif len(categories) == 1:
        category_folder_name = categories[0]
    else:
        category_folder_name = '_'.join(sorted(categories))

    # Multi-shot experiment mode
    if shots_to_test and len(shots_to_test) > 1:
        print(f"\n{'='*70}")
        print(f"Running multi-shot experiments: {shots_to_test}")
        print(f"{'='*70}\n")
        
        # Create multi-shot results directory with category name
        script_dir = Path(__file__).parent
        sam_suffix = "_sam" if args.sam_checkpoint else "_nosam"
        shots_str = '_'.join(map(str, shots_to_test))
        if 'all' in args.category:
            category_str = 'all_categories'
        elif len(args.category) == 1:
            category_str = args.category[0]
        else:
            category_str = '_'.join(args.category)
        multi_shot_dir = script_dir / args.output_dir / f"{category_str}_multi_shot_{shots_str}{sam_suffix}"
        multi_shot_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results for all shots
        all_shots_metrics = {}
        
        for n_shots in shots_to_test:
            print(f"\n{'='*70}")
            print(f"Testing with {n_shots} shots")
            print(f"{'='*70}\n")
            
            shots_metrics = {}
            for category in categories:
                try:
                    # Pass modified output_dir to save inside multi_shot folder
                    multi_shot_output = str(Path(args.output_dir) / f"{category_str}_multi_shot_{shots_str}{sam_suffix}")
                    
                    metrics = run_sam_enhanced_anomaly_detection(
                        category=category,
                        root_dir=args.root_dir,
                        model_name=args.model_name,
                        batch_size=args.batch_size,
                        image_size=args.image_size,
                        use_patches=not args.use_cls,
                        k_neighbors=args.k_neighbors,
                        output_dir=multi_shot_output,
                        visualize_samples=args.visualize_samples,
                        sam_checkpoint=args.sam_checkpoint,
                        sam_model_type=sam_model_type if args.sam_checkpoint else "vit_l",
                        use_sam_for_normal_masking=args.use_sam_masking,
                        few_shot_mode=True,
                        n_shots=n_shots,
                        save_visualizations=args.save_visualizations,
                    )
                    shots_metrics[category] = metrics
                except Exception as e:
                    print(f"Error processing {category} with {n_shots} shots: {e}")
                    continue
            
            all_shots_metrics[n_shots] = shots_metrics
        
        # Save combined results
        combined_file = multi_shot_dir / "all_shots_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_shots_metrics, f, indent=2)
        print(f"\nSaved combined results to {combined_file}")
        
        # Note: To generate comparison plots, run Analysis_and_Plots.py with:
        # generate_shots_comparison_plot(input_dir="{multi_shot_dir}")
        print(f"\nTo generate comparison plots, use Analysis_and_Plots.py:")
        print(f"  generate_shots_comparison_plot(input_dir=r'{multi_shot_dir}')")
        
        # Print summary table
        print(f"\n{'='*70}")
        print("Multi-Shot Experiment Summary:")
        print(f"{'='*70}")
        print(f"{'Shots':<10} {'Category':<15} {'AUROC (Cos)':<15} {'AUROC (Euc)':<15}")
        print(f"{'-'*70}")
        for n_shots in shots_to_test:
            for category in categories:
                if category in all_shots_metrics[n_shots]:
                    m = all_shots_metrics[n_shots][category]
                    auroc_cos = m['cosine']['auroc']
                    auroc_euc = m['euclidean']['auroc']
                    print(f"{n_shots:<10} {category:<15} {auroc_cos:<15.4f} {auroc_euc:<15.4f}")
        print(f"{'='*70}\n")
        
    else:
        # Single shot value mode (original behavior)
        n_shots_value = shots_to_test[0] if shots_to_test else args.n_shots
        all_metrics = {}
        
        # Create single output folder for all categories
        script_dir = Path(__file__).parent
        sam_suffix = "_sam" if args.sam_checkpoint else "_nosam"
        shots_suffix = f"_fewshot_{n_shots_value}" if args.few_shot else "_zeroshot"
        single_output_dir = script_dir / args.output_dir / f"{category_folder_name}{sam_suffix}{shots_suffix}"
        single_output_dir.mkdir(parents=True, exist_ok=True)

        for category in categories:
            try:
                # Create category subdirectory within the main output folder
                category_output_dir = str(single_output_dir / category)
                
                metrics = run_sam_enhanced_anomaly_detection(
                    category=category,
                    root_dir=args.root_dir,
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    use_patches=not args.use_cls,
                    k_neighbors=args.k_neighbors,
                    output_dir=category_output_dir,
                    visualize_samples=args.visualize_samples,
                    sam_checkpoint=args.sam_checkpoint,
                    sam_model_type=sam_model_type if args.sam_checkpoint else "vit_l",
                    use_sam_for_normal_masking=args.use_sam_masking,
                    few_shot_mode=args.few_shot,
                    n_shots=n_shots_value,
                    save_visualizations=args.save_visualizations,
                )
                all_metrics[category] = metrics
            except Exception as e:
                print(f"Error processing {category}: {e}")
                continue

        # Print summary if multiple categories
        if len(categories) > 1:
            print(f"\n{'='*100}")
            print("Summary of Results (Cosine Similarity):")
            print(f"{'='*100}")
            print(f"{'Category':<15} {'AUROC':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Avg Prec':<12}")
            print(f"{'-'*100}")
            for category, metrics in all_metrics.items():
                m = metrics['cosine']
                print(f"{category:<15} {m['auroc']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1_score']:<12.4f} {m['average_precision']:<12.4f}")

            # Compute averages
            avg_auroc = np.mean([m['cosine']['auroc'] for m in all_metrics.values()])
            avg_precision = np.mean([m['cosine']['precision'] for m in all_metrics.values()])
            avg_recall = np.mean([m['cosine']['recall'] for m in all_metrics.values()])
            avg_f1 = np.mean([m['cosine']['f1_score'] for m in all_metrics.values()])
            avg_ap = np.mean([m['cosine']['average_precision'] for m in all_metrics.values()])
            print(f"{'-'*100}")
            print(f"{'MEAN':<15} {avg_auroc:<12.4f} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f} {avg_ap:<12.4f}")
            print(f"{'='*100}\n")

            # Save summary in the output folder
            summary_file = single_output_dir / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"Saved summary to {summary_file}")
            
            # Save metrics to CSV for easy analysis
            metrics_csv_file = single_output_dir / "metrics_summary.csv"
            with open(metrics_csv_file, 'w') as f:
                f.write("Category,Metric,AUROC,Precision,Recall,F1_Score,Average_Precision\n")
                for category, metrics in all_metrics.items():
                    for metric_name in ['cosine', 'euclidean', 'knn']:
                        m = metrics[metric_name]
                        f.write(f"{category},{metric_name},{m['auroc']:.4f},{m['precision']:.4f},{m['recall']:.4f},{m['f1_score']:.4f},{m['average_precision']:.4f}\n")
                # Write averages
                for metric_name in ['cosine', 'euclidean', 'knn']:
                    avg_auroc = np.mean([m[metric_name]['auroc'] for m in all_metrics.values()])
                    avg_precision = np.mean([m[metric_name]['precision'] for m in all_metrics.values()])
                    avg_recall = np.mean([m[metric_name]['recall'] for m in all_metrics.values()])
                    avg_f1 = np.mean([m[metric_name]['f1_score'] for m in all_metrics.values()])
                    avg_ap = np.mean([m[metric_name]['average_precision'] for m in all_metrics.values()])
                    f.write(f"MEAN,{metric_name},{avg_auroc:.4f},{avg_precision:.4f},{avg_recall:.4f},{avg_f1:.4f},{avg_ap:.4f}\n")
            print(f"Saved metrics CSV to {metrics_csv_file}")


if __name__ == "__main__":
    main()


