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
import seaborn as sns

# Import prompt tuning module (optional)
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent / "prompt_based_feature_adaption"))
    from prompt_model import VisualPromptTuning
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    print("Warning: prompt_model not found - prompt tuning will not be available")

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
    
    # Convert to torch tensors and conditional for NVIDIA GPU, Apple Silicon or cpu
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
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
    
    # Convert to torch tensors and conditional for NVIDIA GPU, Apple Silicon or cpu
    if torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
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
            device: Device to use ('cuda', 'mps', or 'cpu')
        """
        self.sam_available = SAM_AVAILABLE and sam_checkpoint is not None
        self.predictor = None
        
        # Optimize device selection for M2 MacBook
        if device == "mps" and torch.backends.mps.is_available():
            self.device = "mps"
        elif device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
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
    
    def get_bbox_from_anomaly_map(self, anomaly_map: np.ndarray, threshold_percentile: float = 60) -> Tuple[int, int, int, int]:
        """
        Extract bounding box from anomaly heatmap.
        
        Args:
            anomaly_map: 2D anomaly heatmap
            threshold_percentile: Percentile for thresholding (lower = more area captured)
            
        Returns:
            Bounding box (x_min, y_min, x_max, y_max)
        """
        # Threshold the map
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
            
            # Use point prompts from heatmap peaks if available
            if anomaly_map is not None:
                # Get top hotspot locations from anomaly map
                threshold = np.percentile(anomaly_map, 90)  # Top 10% hottest areas
                hot_mask = anomaly_map > threshold
                
                # Find peak coordinates
                coords = np.where(hot_mask)
                if len(coords[0]) > 0:
                    # Sample up to 5 points from hottest regions
                    n_points = min(5, len(coords[0]))
                    # Get indices of hottest points
                    flat_indices = np.argsort(anomaly_map[coords[0], coords[1]])[-n_points:]
                    point_coords = np.array([[coords[1][i], coords[0][i]] for i in flat_indices])
                    point_labels = np.ones(len(point_coords), dtype=int)  # All positive prompts
                    
                    # Use both points and bbox for better segmentation
                    input_box = np.array([bbox])
                    masks, scores, logits = self.predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=input_box,
                        multimask_output=True,  # Get multiple options
                    )
                    
                    # Select mask that best matches the anomaly region
                    # Prefer smaller, more focused masks
                    best_idx = np.argmin([m.sum() for m in masks])  # Smallest mask
                    mask = masks[best_idx]
                    
                    # Apply morphological operations to clean up noise
                    mask = self._clean_mask(mask)
                    return mask
            
            # Fallback to bbox-only if no heatmap
            input_box = np.array([bbox])
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=True,
            )
            # Choose smallest mask to avoid over-segmentation
            best_idx = np.argmin([m.sum() for m in masks])
            mask = masks[best_idx]
            
            # Apply morphological operations to clean up noise
            mask = self._clean_mask(mask)
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

class FlippedDataset(torch.utils.data.Dataset):
    """Wrapper to return horizontally flipped images."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Handle subset or original dataset
        if hasattr(self.dataset, 'dataset'):
             # If it's a Subset
             sample = self.dataset.dataset[self.dataset.indices[idx]]
        else:
             sample = self.dataset[idx]
             
        # Create a shallow copy to avoid modifying the original cache
        sample = sample.copy()
        # Flip image tensor (C, H, W) -> Flip on last dimension (W)
        sample['image'] = torch.flip(sample['image'], dims=[-1])
        return sample

class MirroringAnomalyDetector:
    """
    Implements the Mirroring DINO strategy: comparing an image with its flipped version.
    """
    def __init__(self, extractor, image_size=224, patch_size=14):
        self.extractor = extractor
        self.image_size = image_size
        self.patch_size = patch_size
        # Grid size: For patch_size=14 on 224x224 image, we get 224/14=16 patches per side
        # BUT the actual spatial grid is 14x14=196 patches (not 16x16=256)
        # This is because DINOv3 uses overlapping patches or different stride
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
        expected_patches = self.grid_h * self.grid_w  # 14*14 = 196 for DINOv3
        
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
                print(f"Using fallback: returning original embeddings")
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
        
        # 1. Get Original Embeddings
        orig_results = self.extractor.extract_embeddings_batch(
            train_loader, extract_patches=True, extract_cls=False
        )
        orig_patches = orig_results['patch_embeddings']
        
        # 2. Get Flipped Embeddings
        # Create a flipped version of the dataset/loader
        train_dataset_flipped = FlippedDataset(train_loader.dataset)
        flipped_loader = DataLoader(
            train_dataset_flipped, 
            batch_size=train_loader.batch_size, 
            shuffle=False, 
            collate_fn=train_loader.collate_fn
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
        
        self.baseline_mean = np.mean(image_scores)
        self.baseline_std = np.std(image_scores)
        
        print(f"  Normal Baseline: Mean={self.baseline_mean:.4f}, Std={self.baseline_std:.4f}")

    def predict(self, test_loader):
        """
        Evaluation phase: Compute symmetry error for test images.
        """
        print("Evaluating with Mirroring Strategy...")
        
        # 1. Original
        orig_results = self.extractor.extract_embeddings_batch(
            test_loader, extract_patches=True, extract_cls=False
        )
        orig_patches = orig_results['patch_embeddings']
        
        # 2. Flipped
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
        
        # 3. Align
        flip_patches_aligned = self._align_flipped_patches(flip_patches)
        
        # 4. Compute Scores
        orig_norm = orig_patches / (np.linalg.norm(orig_patches, axis=-1, keepdims=True) + 1e-8)
        flip_norm = flip_patches_aligned / (np.linalg.norm(flip_patches_aligned, axis=-1, keepdims=True) + 1e-8)
        
        sim_per_patch = (orig_norm * flip_norm).sum(axis=-1)
        dist_per_patch = 1 - sim_per_patch
        
        # Image-level anomaly score
        raw_scores = dist_per_patch.max(axis=1)
        
        return raw_scores, dist_per_patch, orig_results['labels']


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


def evaluate_anomaly_detection(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    threshold: Optional[float] = None
) -> dict:
    """
    Evaluate anomaly detection performance.

    Args:
        anomaly_scores: Anomaly scores (N,)
        labels: Ground truth labels (N,) - 0 for normal, 1 for anomaly
        threshold: Classification threshold (if None, uses optimal threshold from ROC)

    Returns:
        Dictionary with evaluation metrics including confusion matrix values
    """
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
    confusion_matrix: np.ndarray,
    save_path: Path,
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> None:
    """
    Create and save a confusion matrix visualization.
    
    Args:
        confusion_matrix: 2x2 confusion matrix array
        save_path: Path to save the plot
        class_names: Names for the classes (default: ['Normal', 'Anomaly'])
        title: Plot title
        normalize: Whether to normalize the values
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


def visualize_sam_enhanced_results(
    image: torch.Tensor,
    anomaly_score: float,
    anomaly_heatmap: Optional[np.ndarray],
    sam_mask: Optional[np.ndarray],
    label: int,
    defect_type: str,
    save_path: Path = Path(__file__).parent,
    bbox: Optional[Tuple[int, int, int, int]] = None,
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
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Determine number of subplots based on what's available
    # If no SAM mask, we need 3 plots: original image, heatmap, heatmap with bbox
    # If SAM mask is available, we need 4 plots: original, heatmap, SAM mask, combined
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
        image, anomaly_score, None, None, label, defect_type, save_path, None
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
    use_mirroring: bool = False,
    use_prompts: bool = False,
    prompt_checkpoint: Optional[str] = None,
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
        save_visualizations: Whether to save visualizations
        use_mirroring: Whether to use mirroring for anomaly scoring
    """
    mode_str = "Mirroring" if use_mirroring else (f"Few-Shot ({n_shots})" if few_shot_mode else "Zero-Shot")
    print(f"\n{'='*70}")
    print(f"Run: {mode_str} Anomaly Detection: {category}")
    print(f"{'='*70}\n")

    # Setup output directory - make it relative to script location
    script_dir = Path(__file__).parent
    sam_suffix = "_sam" if sam_checkpoint else "_nosam"
    if use_mirroring:
        method_suffix = "_mirroring"
    else:
        method_suffix = f"_fewshot_{n_shots}" if few_shot_mode else "_zeroshot"
    prompts_suffix = "_prompts" if use_prompts else ""
    output_dir = script_dir / output_dir / f"{category}{sam_suffix}{method_suffix}{prompts_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize SAM with M2 MacBook optimization
    if torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    sam_integrator = SAMIntegrator(sam_checkpoint, model_type=sam_model_type, device=device)

    # Initialize DINOv3 model
    print(f"Loading model: {model_name}")
    extractor = DINOv3EmbeddingExtractor(
        model_name=model_name,
        use_huggingface=True,
    )

    # Load prompt tuning if requested
    prompt_model = None
    if use_prompts:
        if not PROMPTS_AVAILABLE:
            print("Error: Prompt tuning requested but prompt_model not available")
            print("Make sure prompt_based_feature_adaption module is accessible")
            return

        if not prompt_checkpoint:
            print("Error: --use-prompts requires --prompt-checkpoint to be specified")
            return

        if not Path(prompt_checkpoint).exists():
            print(f"Error: Prompt checkpoint not found: {prompt_checkpoint}")
            return

        print(f"Loading prompt checkpoint from {prompt_checkpoint}...")
        checkpoint = torch.load(prompt_checkpoint, map_location='cpu', weights_only=False)

        # Extract prompt configuration from checkpoint
        num_prompts = checkpoint.get('num_prompts', 10)
        embed_dim = checkpoint['prompts'].shape[1]

        # Wrap extractor model with VisualPromptTuning
        prompt_model = VisualPromptTuning(
            dinov3_model=extractor.model,
            num_prompts=num_prompts,
            embed_dim=embed_dim,
        )

        # Load trained prompts
        prompt_model.prompts.data = checkpoint['prompts']
        prompt_model.eval()

        # Attach prompt model to extractor (don't replace the model itself)
        extractor.prompt_model = prompt_model

        print(f"✓ Loaded {num_prompts} prompt tokens (dim={embed_dim})")
        print(f"  Category: {checkpoint.get('category', 'unknown')}")
        print(f"  Original model: {checkpoint.get('model_name', 'unknown')}")

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
    train_loader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=custom_collate_fn,
    )
    print(f"\nLoading test data...")
    test_dataset = MVTecADDataset(root=root_dir, category=category, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    # Initialize variables that may be set differently based on mirroring
    patch_level_scores = None
    
    # Apply Mirroring strategy if enabled
    if use_mirroring:
        print("Using Mirroring DINO strategy...")
        # Initialize Detector (Patch size 14 for ViT-S/14, 16 for others. Adjust if needed)
        p_size = 14 if 'vits14' in model_name or 'dinov2' in model_name else 16
        if 'dinov3' in model_name: p_size = 14  # DINOv3 uses 14
        
        mirroring_detector = MirroringAnomalyDetector(extractor, image_size=image_size, patch_size=p_size)
        
        # "Train" (Estimate Baseline)
        mirroring_detector.fit(train_loader)
        
        # Predict
        anomaly_scores, patch_level_scores, labels_np = mirroring_detector.predict(test_loader)
        test_labels = labels_np
        
        # Create test_embeddings dict for compatibility with evaluation code
        test_embeddings = {'labels': labels_np}
        
        # Set dictionaries for compatibility with evaluation code below
        anomaly_scores_cosine = anomaly_scores
        anomaly_scores_euclidean = anomaly_scores # Placeholder
        anomaly_scores_knn = anomaly_scores # Placeholder
        k_cosine = 1
        k_euclidean = 1
        k_knn = 1
        
    else:
        # Regular (non-mirroring) processing logic
        
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
            print("Applying SAM post-processing...")
            
            for i in range(len(anomaly_scores)):
                # Get test image
                sample = test_dataset[i]
                img_tensor = sample['image']
                
                # Convert to numpy for SAM
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_denorm = img_tensor * std + mean
                img_np = (img_denorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Generate SAM mask using the pre-computed bbox and heatmap
                sam_mask = sam_integrator.predict_mask_from_bbox(img_np, bboxes[i], anomaly_heatmaps[i])
                sam_masks.append(sam_mask)
                
            print(f"Generated {len(sam_masks)} SAM masks")
        else:
            print("SAM not available - heatmaps generated with bounding boxes")
    elif not save_visualizations:
        print("Skipping heatmap generation (--save-visualizations not enabled)")
    
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
    print(f"\n  Euclidean Distance (k={k_euclidean}):")
    print(f"    AUROC: {metrics_euclidean['auroc']:.4f}")
    print(f"    Average Precision: {metrics_euclidean['average_precision']:.4f}")
    print(f"\n  k-NN Distance (k={k_knn}):")
    print(f"    AUROC: {metrics_knn['auroc']:.4f}")
    print(f"    Average Precision: {metrics_knn['average_precision']:.4f}")
    print(f"{'='*70}\n")

    # Save metrics
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Create and save confusion matrix visualizations
    if save_visualizations:
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
        default=str(Path(__file__).parent.parent.parent / "mvtec_ad"),
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
    parser.add_argument(
        "--use-mirroring", 
        action="store_true", 
        help="Enable Mirroring DINO strategy (Compare image vs flipped)"
    )
    parser.add_argument(
        "--use-prompts",
        action="store_true",
        help="Enable prompt-based feature adaptation"
    )
    parser.add_argument(
        "--prompt-checkpoint",
        type=str,
        default=None,
        help="Path to trained prompt checkpoint (e.g., checkpoints/screw_prompts.pt)"
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
            # Check in sam_models directory first, then script directory
            root_dir = Path(__file__).parent.parent.parent
            sam_models_dir = root_dir / "sam_models"
            script_dir = Path(__file__).parent
            
            # Check for SAM checkpoints in order of preference (largest/best first)
            sam_checkpoints = [
                # First check sam_models directory
                sam_models_dir / "sam_vit_h_4b8939.pth",
                sam_models_dir / "sam_vit_l_0b3195.pth",
                sam_models_dir / "sam_vit_b_01ec64.pth",
                # Then check script directory
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
                        use_prompts=args.use_prompts,
                        prompt_checkpoint=args.prompt_checkpoint,
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

        for category in categories:
            try:
                metrics = run_sam_enhanced_anomaly_detection(
                    category=category,
                    root_dir=args.root_dir,
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    image_size=args.image_size,
                    use_patches=not args.use_cls,
                    k_neighbors=args.k_neighbors,
                    output_dir=args.output_dir,
                    visualize_samples=args.visualize_samples,
                    sam_checkpoint=args.sam_checkpoint,
                    sam_model_type=sam_model_type if args.sam_checkpoint else "vit_l",
                    use_sam_for_normal_masking=args.use_sam_masking,
                    few_shot_mode=args.few_shot,
                    n_shots=n_shots_value,
                    save_visualizations=args.save_visualizations,
                    use_mirroring=args.use_mirroring,
                    use_prompts=args.use_prompts,
                    prompt_checkpoint=args.prompt_checkpoint,
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
                print(f"{category:<15} {metrics['cosine']['auroc']:<10.4f} {metrics['cosine']['average_precision']:<10.4f}")

            # Compute average
            avg_auroc = np.mean([m['cosine']['auroc'] for m in all_metrics.values()])
            avg_ap = np.mean([m['cosine']['average_precision'] for m in all_metrics.values()])
            print(f"{'-'*70}")
            print(f"{'Average':<15} {avg_auroc:<10.4f} {avg_ap:<10.4f}")
            print(f"{'='*70}\n")

            # Save summary
            summary_file = Path(__file__).parent / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()


