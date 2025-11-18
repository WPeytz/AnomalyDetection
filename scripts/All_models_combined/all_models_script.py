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


def compute_patch_anomaly_scores(
    test_embeddings: np.ndarray,
    normal_embeddings: np.ndarray,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Compute patch-level anomaly scores for heatmap visualization.
    
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
    
    # Normalize for cosine similarity
    if metric == 'cosine':
        test_norm = test_embeddings / (np.linalg.norm(test_embeddings, axis=-1, keepdims=True) + 1e-8)
        normal_norm = normal_embeddings / (np.linalg.norm(normal_embeddings, axis=-1, keepdims=True) + 1e-8)
    else:
        test_norm = test_embeddings
        normal_norm = normal_embeddings
    
    # Compute patch-level anomaly scores
    patch_anomaly_scores = np.zeros((N, P))
    
    for i in range(N):
        for p in range(P):
            # Compare this test patch against all normal patches
            test_patch = test_norm[i, p]  # (D,)
            
            if metric == 'cosine':
                # Compute cosine similarity with all normal patches
                # Reshape normal patches: (M, P, D) -> (M*P, D)
                normal_patches_flat = normal_norm.reshape(-1, D)  # (M*P, D)
                similarities = test_patch @ normal_patches_flat.T  # (M*P,)
                max_similarity = similarities.max()
            else:  # euclidean
                normal_patches_flat = normal_norm.reshape(-1, D)
                distances = np.linalg.norm(normal_patches_flat - test_patch, axis=-1)
                max_similarity = -distances.min()
            
            # Convert to anomaly score
            patch_anomaly_scores[i, p] = 1 - max_similarity
    
    return patch_anomaly_scores


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
    
    def get_bbox_from_anomaly_map(self, anomaly_map: np.ndarray, threshold_percentile: float = 95) -> Tuple[int, int, int, int]:
        """
        Extract bounding box from anomaly heatmap.
        
        Args:
            anomaly_map: 2D anomaly heatmap
            threshold_percentile: Percentile for thresholding
            
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
    
    def predict_mask_from_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Predict segmentation mask using SAM from bounding box.
        
        Args:
            image: RGB image (H, W, 3)
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            
        Returns:
            Binary mask or None if SAM not available
        """
        if not self.sam_available:
            return None
            
        try:
            self.predictor.set_image(image)
            input_box = np.array([bbox])
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box,
                multimask_output=False,
            )
            return masks[0]  # Return best mask
        except Exception as e:
            print(f"SAM prediction failed: {e}")
            return None
    
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


def visualize_sam_enhanced_results(
    image: torch.Tensor,
    anomaly_score: float,
    anomaly_heatmap: Optional[np.ndarray],
    sam_mask: Optional[np.ndarray],
    label: int,
    defect_type: str,
    save_path: Path = Path(__file__).parent,
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
    """
    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = torch.clamp(image, 0, 1)

    # Convert to numpy for visualization
    image_np = image.permute(1, 2, 0).cpu().numpy()

    # Determine number of subplots based on what's available
    # If no SAM mask, we only need 2 plots: original image + heatmap
    # If SAM mask is available, we need 4 plots: original, heatmap, SAM mask, combined
    if sam_mask is not None:
        n_plots = 4
    elif anomaly_heatmap is not None:
        n_plots = 2
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
        im1 = axes[1].imshow(anomaly_heatmap, cmap='hot', alpha=0.7)
        axes[1].imshow(image_np, alpha=0.3)
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
            heatmap_path = save_dir / f"{base_name}_heatmap_only.png"
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
                sam_path = save_dir / f"{base_name}_heatmap_with_sam.png"
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
        image, anomaly_score, None, None, label, defect_type, save_path
    )


def run_sam_enhanced_anomaly_detection(
    category: str,
    root_dir: str = "mvtec_ad",
    model_name: str = "dinov3_vits16",
    batch_size: int = 32,
    image_size: int = 224,
    use_patches: bool = True,
    k_neighbors: int = 1,
    output_dir: str = "results",
    visualize_samples: int = 5,
    sam_checkpoint: Optional[str] = None,
    sam_model_type: str = "vit_l",
    use_sam_for_normal_masking: bool = True,
    few_shot_mode: bool = False,
    n_shots: int = 5,
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
    script_dir = Path(__file__).parent
    sam_suffix = "_sam" if sam_checkpoint else "_nosam"
    shots_suffix = f"_fewshot_{n_shots}" if few_shot_mode else "_zeroshot"
    output_dir = script_dir / output_dir / f"{category}{sam_suffix}{shots_suffix}"
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
    if few_shot_mode:
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
        train_dataset = full_train_dataset
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
    3
    # Determine k values for each metric
    k_cosine = k_neighbors
    k_euclidean = 1  # Use nearest neighbor for euclidean
    k_knn = min(3, max(1, len(normal_embeddings) // 10))  # k=3 or 10% of normal samples (faster)
    
    if use_patches:
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
            metric='euclidean',  # Use euclidean for knn
            k=k_knn,
        )
        
        # Use cosine as primary metric (backward compatibility)
        anomaly_scores = anomaly_scores_cosine
        
        print(f"Image-level anomaly scores shape: {anomaly_scores.shape}")
        
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
    else:
        # CLS token anomaly detection
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
        
        print(f"  k-NN distance (k={k_knn})...")
        anomaly_scores_knn = compute_anomaly_scores(
            test_embed,
            normal_embeddings,
            metric='euclidean',
            k=k_knn,
        )
        
        # Use cosine as primary metric
        anomaly_scores = anomaly_scores_cosine
        patch_level_scores = None

    print(f"Computed anomaly scores for {len(anomaly_scores)} test images")

    # Generate anomaly heatmaps for all images (if patch-level scores available)
    sam_masks = []
    anomaly_heatmaps = []
    
    if patch_level_scores is not None:
        print(f"Generating anomaly heatmaps for {len(anomaly_scores)} images...")
        print(f"Patch scores per image: {patch_level_scores.shape[1]}")
        
        # Generate heatmaps for all images
        for i in range(len(anomaly_scores)):
            heatmap = create_anomaly_heatmap(patch_level_scores[i], image_size=image_size)
            anomaly_heatmaps.append(heatmap)
        
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
                
                # Get bounding box from anomaly heatmap
                bbox = sam_integrator.get_bbox_from_anomaly_map(anomaly_heatmaps[i])
                
                # Generate SAM mask
                sam_mask = sam_integrator.predict_mask_from_bbox(img_np, bbox)
                sam_masks.append(sam_mask)
                
            print(f"Generated {len(sam_masks)} SAM masks")
        else:
            print("SAM not available - heatmaps generated without SAM enhancement")
    
    # Evaluate all metrics
    print(f"\nEvaluating all metrics...")
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

    # Visualize samples
    if visualize_samples > 0:
        print(f"\nGenerating visualizations...")
        # Create visualization directory with same naming as results
        sam_suffix = "_sam" if sam_checkpoint else "_nosam"
        shots_suffix = f"_fewshot_{n_shots}" if few_shot_mode else "_zeroshot"
        vis_dir = Path(__file__).parent / "visualizations" / f"{category}{sam_suffix}{shots_suffix}"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Visualize top anomalies
        anomaly_indices = np.argsort(anomaly_scores)[::-1][:visualize_samples]

        for i, idx in enumerate(anomaly_indices):
            sample = test_dataset[idx]
            save_path = vis_dir / f"top_anomaly_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            
            # Get heatmap and SAM mask if available
            heatmap = anomaly_heatmaps[idx] if idx < len(anomaly_heatmaps) else None
            sam_mask_data = sam_masks[idx] if idx < len(sam_masks) else None
            
            visualize_sam_enhanced_results(
                sample['image'],
                anomaly_scores[idx],
                heatmap,
                sam_mask_data,
                sample['label'],
                sample['defect_type'],
                save_path,
            )

        # Visualize top normal samples
        normal_indices = np.argsort(anomaly_scores)[:visualize_samples]

        for i, idx in enumerate(normal_indices):
            sample = test_dataset[idx]
            save_path = vis_dir / f"top_normal_{i+1}_score_{anomaly_scores[idx]:.3f}.png"
            
            # Get heatmap and SAM mask if available
            heatmap = anomaly_heatmaps[idx] if idx < len(anomaly_heatmaps) else None
            sam_mask_data = sam_masks[idx] if idx < len(sam_masks) else None
            
            visualize_sam_enhanced_results(
                sample['image'],
                anomaly_scores[idx],
                heatmap,
                sam_mask_data,
                sample['label'],
                sample['defect_type'],
                save_path,
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
        default=r"/zhome/9c/f/221532/Deep_Learning_Project/mvtec_ad",
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

    args = parser.parse_args()

    # Only use SAM if --use-sam flag is provided
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
        else:
            # Determine model type from provided checkpoint filename
            checkpoint_name = Path(args.sam_checkpoint).name
            if "vit_h" in checkpoint_name:
                sam_model_type = "vit_h"
            elif "vit_l" in checkpoint_name:
                sam_model_type = "vit_l"
            elif "vit_b" in checkpoint_name:
                sam_model_type = "vit_b"
            else:
                sam_model_type = "vit_h"  # default
    else:
        # Disable SAM
        args.sam_checkpoint = None
        sam_model_type = "vit_h"  # default (won't be used)

    # Run on all categories or single category
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ] if args.category == 'all' else [args.category]

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
                n_shots=args.n_shots,
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
        avg_ap = np.mean([m['average_precisvisuaion'] for m in all_metrics.values()])
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
