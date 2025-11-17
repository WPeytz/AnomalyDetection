"""
DINOv3 Embedding Extraction Pipeline for MVTec AD

This module provides utilities to extract embeddings from images using
pretrained DINOv3 models for anomaly detection tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path


class DINOv3EmbeddingExtractor:
    """
    Extract embeddings from images using pretrained DINOv3 models.

    Supports both:
    1. torch.hub loading (requires local repo and weights)
    2. Hugging Face transformers (easier, automatic download)

    Args:
        model_name: Name of DINOv3 model (e.g., 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16')
        device: Device to run model on ('cuda' or 'cpu')
        use_huggingface: If True, use HuggingFace transformers instead of torch.hub
    """

    def __init__(
        self,
        model_name: str = 'dinov3_vits16',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_huggingface: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.use_huggingface = use_huggingface
        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):
        """Load pretrained DINOv3 model."""
        print(f"Loading {self.model_name} on {self.device}...")

        if self.use_huggingface:
            self._load_huggingface_model()
        else:
            self._load_torchhub_model()

        self.model.eval()
        print(f"✓ Model loaded successfully!")

    def _load_huggingface_model(self):
        """Load model using Hugging Face transformers."""
        from transformers import AutoImageProcessor, AutoModel

        # Map model names to HuggingFace model IDs
        hf_model_map = {
            'dinov3_vits16': 'facebook/dinov3-vits16-pretrain-lvd1689m',
            'dinov3_vits16plus': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
            'dinov3_vitb16': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_vitl16': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'dinov3_vith16plus': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
            'dinov3_vit7b16': 'facebook/dinov3-vit7b16-pretrain-lvd1689m',
        }

        if self.model_name not in hf_model_map:
            raise ValueError(f"Model {self.model_name} not found in HuggingFace map. "
                           f"Available models: {list(hf_model_map.keys())}")

        model_id = hf_model_map[self.model_name]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, device_map=self.device)

    def _load_torchhub_model(self):
        """Load model using torch.hub (requires local repo)."""
        # This assumes you have downloaded the model weights
        # and are running from the dinov3 repository
        repo_dir = Path(__file__).parent.parent
        self.model = torch.hub.load(
            str(repo_dir),
            self.model_name,
            source='local',
            pretrained=True
        )
        self.model = self.model.to(self.device)

    @torch.no_grad()
    def extract_patch_embeddings(
        self,
        image: torch.Tensor,
        return_cls_token: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract patch-level embeddings from an image.

        Args:
            image: Input image tensor (B, C, H, W) or (C, H, W)
            return_cls_token: Whether to return CLS token embedding

        Returns:
            Dictionary containing:
                - cls_token: CLS token embedding (B, D) if return_cls_token=True
                - patch_embeddings: Patch embeddings (B, N, D) where N is number of patches
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        if self.use_huggingface:
            outputs = self.model(pixel_values=image, output_hidden_states=True)
            # For HuggingFace models
            # last_hidden_state has shape (B, N+1, D) where N is num patches
            last_hidden = outputs.last_hidden_state
            cls_token = outputs.pooler_output if hasattr(outputs, 'pooler_output') else last_hidden[:, 0]
            patch_embeddings = last_hidden[:, 1:]  # Exclude CLS token
        else:
            # For torch.hub models
            features = self.model.forward_features(image)
            # DINOv3 returns dict with 'x_norm_clstoken' and 'x_norm_patchtokens'
            if isinstance(features, dict):
                cls_token = features['x_norm_clstoken']
                patch_embeddings = features['x_norm_patchtokens']
            else:
                # Fallback if model returns tensor
                cls_token = features[:, 0]
                patch_embeddings = features[:, 1:]

        result = {'patch_embeddings': patch_embeddings}
        if return_cls_token:
            result['cls_token'] = cls_token

        return result

    @torch.no_grad()
    def extract_embeddings_batch(
        self,
        dataloader: DataLoader,
        extract_patches: bool = True,
        extract_cls: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for an entire dataset.

        Args:
            dataloader: PyTorch DataLoader for the dataset
            extract_patches: Whether to extract patch embeddings
            extract_cls: Whether to extract CLS token embeddings

        Returns:
            Dictionary containing:
                - cls_embeddings: CLS token embeddings (N, D)
                - patch_embeddings: Patch embeddings (N, P, D) if extract_patches=True
                - labels: Labels for each sample
        """
        cls_embeddings_list = []
        patch_embeddings_list = []
        labels_list = []

        print("Extracting embeddings...")
        for batch in tqdm(dataloader):
            images = batch['image']
            labels = batch['label']

            embeddings = self.extract_patch_embeddings(images, return_cls_token=extract_cls)

            if extract_cls:
                cls_embeddings_list.append(embeddings['cls_token'].cpu().numpy())

            if extract_patches:
                patch_embeddings_list.append(embeddings['patch_embeddings'].cpu().numpy())

            labels_list.append(labels.numpy())

        result = {
            'labels': np.concatenate(labels_list, axis=0)
        }

        if extract_cls:
            result['cls_embeddings'] = np.concatenate(cls_embeddings_list, axis=0)

        if extract_patches:
            result['patch_embeddings'] = np.concatenate(patch_embeddings_list, axis=0)

        return result

    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        if self.use_huggingface:
            return self.model.config.hidden_size
        else:
            return self.model.embed_dim


def compute_similarity_scores(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Compute similarity scores between query and reference embeddings.

    Args:
        query_embeddings: Query embeddings (N, D) or (N, P, D)
        reference_embeddings: Reference embeddings (M, D) or (M, P, D)
        metric: Similarity metric ('cosine' or 'euclidean')

    Returns:
        Similarity scores (N, M) or (N, P, M, P) for patch embeddings
    """
    if metric == 'cosine':
        # Normalize embeddings
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=-1, keepdims=True) + 1e-8)
        reference_norm = reference_embeddings / (np.linalg.norm(reference_embeddings, axis=-1, keepdims=True) + 1e-8)

        if query_embeddings.ndim == 2:
            # CLS token embeddings
            similarities = query_norm @ reference_norm.T
        else:
            # Patch embeddings: (N, P, D) @ (M, P, D)
            # Compute max similarity across patches
            N, P, D = query_embeddings.shape
            M = reference_embeddings.shape[0]

            similarities = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    # Compute pairwise cosine similarity between patches
                    patch_sim = query_norm[i] @ reference_norm[j].T  # (P, P)
                    # Take max similarity for each query patch
                    similarities[i, j] = patch_sim.max(axis=1).mean()

    elif metric == 'euclidean':
        if query_embeddings.ndim == 2:
            # CLS token embeddings
            similarities = -np.linalg.norm(
                query_embeddings[:, None, :] - reference_embeddings[None, :, :],
                axis=-1
            )
        else:
            # Patch embeddings
            N, P, D = query_embeddings.shape
            M = reference_embeddings.shape[0]

            similarities = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    # Compute pairwise euclidean distance between patches
                    dists = np.linalg.norm(
                        query_embeddings[i, :, None, :] - reference_embeddings[j, None, :, :],
                        axis=-1
                    )  # (P, P)
                    # Take min distance for each query patch (negative for similarity)
                    similarities[i, j] = -dists.min(axis=1).mean()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarities


def compute_anomaly_scores(
    test_embeddings: np.ndarray,
    normal_embeddings: np.ndarray,
    metric: str = 'cosine',
    k: int = 1,
) -> np.ndarray:
    """
    Compute anomaly scores based on similarity to normal samples.

    Args:
        test_embeddings: Test embeddings (N, D) or (N, P, D)
        normal_embeddings: Normal embeddings (M, D) or (M, P, D)
        metric: Similarity metric ('cosine' or 'euclidean')
        k: Number of nearest neighbors to consider

    Returns:
        Anomaly scores (N,) - higher means more anomalous
    """
    similarities = compute_similarity_scores(test_embeddings, normal_embeddings, metric)

    # Get k nearest neighbors
    if k == 1:
        max_similarities = similarities.max(axis=1)
    else:
        # Take average of top-k similarities
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:]
        max_similarities = np.mean(
            np.take_along_axis(similarities, top_k_indices, axis=1),
            axis=1
        )

    # Convert similarity to anomaly score (lower similarity = higher anomaly)
    anomaly_scores = 1 - max_similarities

    return anomaly_scores


if __name__ == "__main__":
    # Example usage
    from mvtec_dataset import MVTecADDataset, get_mvtec_transforms

    # Initialize extractor
    extractor = DINOv3EmbeddingExtractor(
        model_name='dinov3_vits16',
        use_huggingface=True,
    )

    # Load dataset
    transform = get_mvtec_transforms(224)
    train_dataset = MVTecADDataset(
        root="./mvtec_ad",
        category="bottle",
        split="train",
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Extract embeddings
    train_embeddings = extractor.extract_embeddings_batch(
        train_loader,
        extract_patches=True,
        extract_cls=True,
    )

    print(f"\nExtracted embeddings:")
    print(f"  CLS embeddings shape: {train_embeddings['cls_embeddings'].shape}")
    print(f"  Patch embeddings shape: {train_embeddings['patch_embeddings'].shape}")
    print(f"  Labels shape: {train_embeddings['labels'].shape}")
    print(f"  Embedding dimension: {extractor.get_embedding_dim()}")
