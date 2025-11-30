"""
Training script for prompt-based feature adaptation.

This script trains visual prompts for DINOv3 to improve anomaly detection
on challenging categories using few-shot learning.
"""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from typing import Tuple, Dict

# Add parent directory to path for imports
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


class SeparabilityLoss(nn.Module):
    """
    Loss function that maximizes separability between normal and defect samples.

    Separability = (μ_NN - μ_ND) / (σ_NN + σ_ND)

    Where:
        μ_NN = mean similarity between normal samples
        μ_ND = mean similarity between normal and defect samples
        σ_NN = std of normal-normal similarities
        σ_ND = std of normal-defect similarities
    """

    def __init__(self, use_patches: bool = True):
        super().__init__()
        self.use_patches = use_patches

    def compute_similarities(self, embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise cosine similarities.

        Args:
            embed1: [N, D] or [N, P, D]
            embed2: [M, D] or [M, P, D]

        Returns:
            Similarities [N, M]
        """
        # Normalize embeddings
        embed1_norm = embed1 / (embed1.norm(dim=-1, keepdim=True) + 1e-8)
        embed2_norm = embed2 / (embed2.norm(dim=-1, keepdim=True) + 1e-8)

        if embed1.dim() == 2:
            # CLS tokens: [N, D] @ [M, D].T -> [N, M]
            similarities = embed1_norm @ embed2_norm.T
        else:
            # Patch embeddings: [N, P, D]
            N, P, D = embed1.shape
            M = embed2.shape[0]

            similarities = torch.zeros(N, M, device=embed1.device)
            for i in range(N):
                for j in range(M):
                    # Max similarity across patches
                    patch_sim = embed1_norm[i] @ embed2_norm[j].T  # [P, P]
                    similarities[i, j] = patch_sim.max(dim=1)[0].mean()

        return similarities

    def forward(
        self,
        normal_embed: torch.Tensor,
        defect_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute separability loss.

        Args:
            normal_embed: Normal sample embeddings [N, D] or [N, P, D]
            defect_embed: Defect sample embeddings [M, D] or [M, P, D]

        Returns:
            loss: Negative separability (to minimize)
            metrics: Dictionary with intermediate metrics
        """
        # Compute similarities
        nn_sim = self.compute_similarities(normal_embed, normal_embed)
        nd_sim = self.compute_similarities(normal_embed, defect_embed)

        # Remove diagonal from normal-normal similarities (self-similarity)
        mask = ~torch.eye(nn_sim.shape[0], dtype=torch.bool, device=nn_sim.device)
        nn_sim_off_diag = nn_sim[mask]

        # Compute statistics
        mu_nn = nn_sim_off_diag.mean()
        mu_nd = nd_sim.mean()
        sigma_nn = nn_sim_off_diag.std()
        sigma_nd = nd_sim.std()

        # Separability metric
        separability = (mu_nn - mu_nd) / (sigma_nn + sigma_nd + 1e-8)

        # Loss: maximize separability = minimize negative separability
        loss = -separability

        metrics = {
            'separability': separability.item(),
            'mu_nn': mu_nn.item(),
            'mu_nd': mu_nd.item(),
            'sigma_nn': sigma_nn.item(),
            'sigma_nd': sigma_nd.item(),
        }

        return loss, metrics


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for anomaly detection.

    Pulls normal samples together and pushes defect samples away.
    """

    def __init__(self, margin: float = 0.5, use_patches: bool = True):
        super().__init__()
        self.margin = margin
        self.use_patches = use_patches

    def compute_similarity(self, embed1: torch.Tensor, embed2: torch.Tensor) -> torch.Tensor:
        """Compute mean cosine similarity between two embeddings."""
        embed1_norm = embed1 / (embed1.norm(dim=-1, keepdim=True) + 1e-8)
        embed2_norm = embed2 / (embed2.norm(dim=-1, keepdim=True) + 1e-8)

        if embed1.dim() == 2:
            # CLS tokens
            similarity = (embed1_norm * embed2_norm).sum(dim=-1)
        else:
            # Patch embeddings: max similarity across patches
            N, P, D = embed1.shape
            M = embed2.shape[0]
            similarities = []
            for i in range(N):
                sim = (embed1_norm[i].unsqueeze(1) * embed2_norm.unsqueeze(0)).sum(dim=-1)
                similarities.append(sim.max(dim=1)[0].mean(dim=0))
            similarity = torch.stack(similarities)

        return similarity

    def forward(
        self,
        normal_embed: torch.Tensor,
        defect_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute contrastive loss.

        Args:
            normal_embed: [N, D] or [N, P, D]
            defect_embed: [M, D] or [M, P, D]

        Returns:
            loss: Contrastive loss
            metrics: Dictionary with intermediate metrics
        """
        # Normal-normal similarity (should be high)
        nn_sim = []
        for i in range(len(normal_embed)):
            for j in range(i + 1, len(normal_embed)):
                nn_sim.append(self.compute_similarity(
                    normal_embed[i:i+1], normal_embed[j:j+1]
                ))
        nn_sim = torch.cat(nn_sim) if nn_sim else torch.tensor([1.0], device=normal_embed.device)

        # Normal-defect similarity (should be low)
        nd_sim = []
        for i in range(len(normal_embed)):
            for j in range(len(defect_embed)):
                nd_sim.append(self.compute_similarity(
                    normal_embed[i:i+1], defect_embed[j:j+1]
                ))
        nd_sim = torch.cat(nd_sim)

        # Contrastive loss
        pos_loss = (1 - nn_sim).mean()  # Pull normal samples together
        neg_loss = torch.relu(nd_sim - self.margin).mean()  # Push defects away

        loss = pos_loss + neg_loss

        metrics = {
            'pos_loss': pos_loss.item(),
            'neg_loss': neg_loss.item(),
            'nn_sim': nn_sim.mean().item(),
            'nd_sim': nd_sim.mean().item(),
        }

        return loss, metrics


def train_prompts(
    model: nn.Module,
    train_loader: DataLoader,
    defect_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    loss_type: str = 'separability',
    device: str = 'cuda',
    use_patches: bool = True,
) -> Dict:
    """
    Train visual prompts using few-shot learning.

    Args:
        model: VisualPromptTuning model
        train_loader: DataLoader for normal samples
        defect_loader: DataLoader for defect samples
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        loss_type: 'separability' or 'contrastive'
        device: Device to train on
        use_patches: Use patch embeddings (vs CLS token)

    Returns:
        Training history dictionary
    """
    # Setup loss function
    if loss_type == 'separability':
        criterion = SeparabilityLoss(use_patches=use_patches)
    elif loss_type == 'contrastive':
        criterion = ContrastiveLoss(use_patches=use_patches)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Only optimize prompts (and adapters if present)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )

    # Training history
    history = {
        'loss': [],
        'separability': [],
        'mu_nn': [],
        'mu_nd': [],
    }

    model.train()
    print(f"\nTraining with {loss_type} loss for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_metrics = []

        # Iterate through batches
        for normal_batch, defect_batch in zip(train_loader, defect_loader):
            normal_images = normal_batch['image'].to(device)
            defect_images = defect_batch['image'].to(device)

            # Forward pass
            normal_output = model(normal_images, return_cls=not use_patches, return_patches=use_patches)
            defect_output = model(defect_images, return_cls=not use_patches, return_patches=use_patches)

            # Get embeddings
            if use_patches:
                normal_embed = normal_output['patch_embeddings']
                defect_embed = defect_output['patch_embeddings']
            else:
                normal_embed = normal_output['cls_embeddings']
                defect_embed = defect_output['cls_embeddings']

            # Compute loss
            loss, metrics = criterion(normal_embed, defect_embed)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_metrics.append(metrics)

        scheduler.step()

        # Average metrics
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {
            key: np.mean([m[key] for m in epoch_metrics if key in m])
            for key in epoch_metrics[0].keys()
        }

        # Log history
        history['loss'].append(avg_loss)
        for key, value in avg_metrics.items():
            if key not in history:
                history[key] = []
            history[key].append(value)

        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}", end='')
            if 'separability' in avg_metrics:
                print(f" | Separability: {avg_metrics['separability']:.4f}", end='')
            print()

    print("✓ Training completed!")
    return history


def main():
    parser = argparse.ArgumentParser(description="Train visual prompts for DINOv3 anomaly detection")
    parser.add_argument("--category", type=str, required=True,
                       help="MVTec AD category")
    parser.add_argument("--root-dir", type=str, default="mvtec_ad",
                       help="Root directory of MVTec AD dataset")
    parser.add_argument("--model-name", type=str, default="dinov3_vits16",
                       choices=['dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'],
                       help="DINOv3 model variant")
    parser.add_argument("--num-prompts", type=int, default=10,
                       help="Number of prompt tokens")
    parser.add_argument("--num-defect-samples", type=int, default=10,
                       help="Number of defect samples for few-shot learning")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--loss-type", type=str, default="separability",
                       choices=['separability', 'contrastive'],
                       help="Loss function type")
    parser.add_argument("--use-cls", action="store_true",
                       help="Use CLS token instead of patch embeddings")
    parser.add_argument("--use-adapters", action="store_true",
                       help="Use adapter layers in addition to prompts")
    parser.add_argument("--modulation-type", type=str, default="per_patch",
                       choices=['per_patch', 'global', 'learned_transform'],
                       help="Type of prompt modulation")
    parser.add_argument("--scaling-factor", type=float, default=0.1,
                       help="Scaling factor for prompt influence")
    parser.add_argument("--output-dir", type=str, default="prompt_based_feature_adaption/checkpoints",
                       help="Output directory for checkpoints")

    args = parser.parse_args()

    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available()
                         else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading {args.model_name} with {args.num_prompts} prompts...")
    model = load_dinov3_for_prompting(
        model_name=args.model_name,
        num_prompts=args.num_prompts,
        device=device,
        use_adapters=args.use_adapters,
        modulation_type=args.modulation_type,
        scaling_factor=args.scaling_factor,
    )

    # Setup datasets
    transform = get_mvtec_transforms(224)

    print(f"\nLoading MVTec AD: {args.category}")

    # Normal samples (full training set)
    train_dataset = MVTecADDataset(
        root=args.root_dir,
        category=args.category,
        split='train',
        transform=transform,
    )
    print(f"  Normal samples: {len(train_dataset)}")

    # Defect samples (few-shot subset from test set)
    test_dataset = MVTecADDataset(
        root=args.root_dir,
        category=args.category,
        split='test',
        transform=transform,
    )
    defect_indices = test_dataset.get_anomaly_samples()[:args.num_defect_samples]
    defect_dataset = Subset(test_dataset, defect_indices)
    print(f"  Defect samples (few-shot): {len(defect_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    defect_loader = DataLoader(
        defect_dataset,
        batch_size=min(args.batch_size, len(defect_dataset)),
        shuffle=True,
        drop_last=False,
        collate_fn=custom_collate_fn,
    )

    # Train prompts
    history = train_prompts(
        model=model,
        train_loader=train_loader,
        defect_loader=defect_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        device=device,
        use_patches=not args.use_cls,
    )

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / f"{args.category}_prompts.pt"
    checkpoint_data = {
        'model_name': args.model_name,
        'category': args.category,
        'num_prompts': args.num_prompts,
        'prompts': model.prompts.data,
        'prompt_scales': model.prompt_scales.data,
        'modulation_type': args.modulation_type,
        'scaling_factor': args.scaling_factor,
        'history': history,
        'args': vars(args),
    }
    # Save MLP weights if using learned_transform
    if args.modulation_type == 'learned_transform':
        checkpoint_data['transform_mlp_state'] = model.transform_mlp.state_dict()
    torch.save(checkpoint_data, checkpoint_path)

    print(f"\n✓ Checkpoint saved to {checkpoint_path}")

    # Save training history
    history_path = output_dir / f"{args.category}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✓ Training history saved to {history_path}")

    # Print final metrics
    print("\nFinal Metrics:")
    print(f"  Loss: {history['loss'][-1]:.4f}")
    if 'separability' in history:
        print(f"  Separability: {history['separability'][-1]:.4f}")


if __name__ == "__main__":
    main()
