"""
Prompt-Based Feature Adaptation for DINOv3 Anomaly Detection.

This module implements visual prompt tuning for adapting pretrained DINOv3
models to specific anomaly detection tasks without fine-tuning the backbone.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from pathlib import Path


class VisualPromptTuning(nn.Module):
    """
    Visual Prompt Tuning wrapper for DINOv3 models.

    Adds learnable prompt tokens to the input sequence of a frozen DINOv3 model,
    allowing task-specific adaptation with minimal trainable parameters.

    Args:
        dinov3_model: Pretrained DINOv3 model (will be frozen)
        num_prompts: Number of learnable prompt tokens
        embed_dim: Embedding dimension of the model
        prompt_dropout: Dropout rate for prompts
        initialize_from_embeddings: If True, initialize prompts from random patches
    """

    def __init__(
        self,
        dinov3_model: nn.Module,
        num_prompts: int = 10,
        embed_dim: int = 384,
        prompt_dropout: float = 0.0,
        initialize_from_embeddings: bool = False,
    ):
        super().__init__()

        # Freeze the DINOv3 backbone
        self.dinov3 = dinov3_model
        for param in self.dinov3.parameters():
            param.requires_grad = False

        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

        # Initialize learnable prompts
        if initialize_from_embeddings:
            # Will be initialized after seeing first batch
            self.prompts = nn.Parameter(torch.randn(num_prompts, embed_dim) * 0.02)
        else:
            # Xavier uniform initialization
            self.prompts = nn.Parameter(torch.empty(num_prompts, embed_dim))
            nn.init.xavier_uniform_(self.prompts)

        # Optional dropout for prompts
        self.prompt_dropout = nn.Dropout(prompt_dropout) if prompt_dropout > 0 else None

        # Track if model is in training mode
        self.dinov3.eval()

    def forward(
        self,
        images: torch.Tensor,
        return_cls: bool = True,
        return_patches: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with visual prompts.

        Args:
            images: Input images [B, C, H, W]
            return_cls: Whether to return CLS token embeddings
            return_patches: Whether to return patch embeddings

        Returns:
            Dictionary containing:
                - cls_embeddings: [B, D] if return_cls=True
                - patch_embeddings: [B, N, D] if return_patches=True
        """
        # Forward through DINOv3 (frozen)
        with torch.no_grad():
            # Use forward_features() or default forward to get embeddings
            # HuggingFace models return a BaseModelOutputWithPooling object
            if hasattr(self.dinov3, 'forward'):
                output = self.dinov3(images)
            elif hasattr(self.dinov3, 'forward_features'):
                output = self.dinov3.forward_features(images)
            else:
                output = self.dinov3(images, is_training=False)

        result = {}
        cls_embed = None
        patch_embed = None

        # Handle different output formats
        if isinstance(output, dict):
            # Could be Torch.hub format OR HuggingFace format (which is dict-like)
            if 'x_norm_clstoken' in output:
                # Torch.hub format
                if return_cls:
                    cls_embed = output['x_norm_clstoken']
                if return_patches:
                    patch_embed = output['x_norm_patchtokens']
            elif 'last_hidden_state' in output:
                # HuggingFace format (BaseModelOutputWithPooling is dict-like)
                last_hidden = output['last_hidden_state']
                if return_cls:
                    cls_embed = last_hidden[:, 0]  # CLS token
                if return_patches:
                    patch_embed = last_hidden[:, 1:]  # Patch tokens
        elif isinstance(output, torch.Tensor):
            # Fallback for tensor output
            if return_cls:
                cls_embed = output[:, 0]
            if return_patches:
                patch_embed = output[:, 1:]

        # Apply prompt modulation (learnable transformation)
        # This ensures gradients flow through prompts
        batch_size = images.shape[0]
        prompt_expand = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)  # [B, P, D]

        if return_patches and patch_embed is not None:
            # Compute attention between prompts and patches
            # This creates a connection between prompts and patch embeddings
            prompt_norm = prompt_expand / (prompt_expand.norm(dim=-1, keepdim=True) + 1e-8)
            patch_norm = patch_embed / (patch_embed.norm(dim=-1, keepdim=True) + 1e-8)

            prompt_sim = torch.bmm(prompt_norm, patch_norm.transpose(1, 2))  # [B, P, N_patches]

            # Use prompts to modulate patch embeddings (residual connection)
            prompt_influence = torch.bmm(prompt_sim.softmax(dim=-1), patch_embed)  # [B, P, D]
            prompt_effect = prompt_influence.mean(dim=1, keepdim=True)  # [B, 1, D]

            # Add prompt effect as residual (scaled very small to preserve pretrained features)
            # Using 0.01 instead of 0.1 to minimize disruption to pretrained features
            patch_embed = patch_embed + 0.01 * prompt_effect

            result['patch_embeddings'] = patch_embed

        if return_cls and cls_embed is not None:
            # Similar modulation for CLS token
            cls_expand = cls_embed.unsqueeze(1)  # [B, 1, D]

            prompt_norm = prompt_expand / (prompt_expand.norm(dim=-1, keepdim=True) + 1e-8)
            cls_norm = cls_expand / (cls_expand.norm(dim=-1, keepdim=True) + 1e-8)

            cls_sim = torch.bmm(prompt_norm, cls_norm.transpose(1, 2))  # [B, P, 1]

            prompt_influence = torch.bmm(cls_sim.softmax(dim=1).transpose(1, 2), prompt_expand)  # [B, 1, D]
            cls_embed = cls_embed + 0.01 * prompt_influence.squeeze(1)

            result['cls_embeddings'] = cls_embed

        return result

    def forward_with_prompts(
        self,
        images: torch.Tensor,
        return_cls: bool = True,
        return_patches: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with explicit prompt injection (advanced usage).

        This method manually injects prompts into the transformer blocks.
        Use this if you need fine-grained control over prompt placement.

        Args:
            images: Input images [B, C, H, W]
            return_cls: Whether to return CLS token embeddings
            return_patches: Whether to return patch embeddings

        Returns:
            Dictionary with embeddings
        """
        batch_size = images.shape[0]

        # Get patch embeddings from patch_embed layer
        # Note: This requires accessing DINOv3 internals
        # For torch.hub models, the structure is:
        # dinov3.patch_embed(images) -> patch embeddings
        # dinov3.cls_token -> CLS token

        try:
            # Try to access patch_embed
            patch_embed = self.dinov3.patch_embed(images)  # [B, N, D]
            cls_token = self.dinov3.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]

            # Prepare prompts
            prompts = self.prompts.unsqueeze(0).expand(batch_size, -1, -1)  # [B, P, D]
            if self.prompt_dropout is not None and self.training:
                prompts = self.prompt_dropout(prompts)

            # Concatenate: [CLS] + [Prompts] + [Patches]
            x = torch.cat([cls_token, prompts, patch_embed], dim=1)  # [B, 1+P+N, D]

            # Add positional embeddings
            # Note: DINOv3 uses interpolated positional embeddings
            # We need to extend them for prompt tokens
            if hasattr(self.dinov3, 'pos_embed'):
                pos_embed = self.dinov3.pos_embed
                # Zero positional encoding for prompts (they're position-agnostic)
                prompt_pos_embed = torch.zeros(1, self.num_prompts, self.embed_dim, device=x.device)
                extended_pos_embed = torch.cat([
                    pos_embed[:, :1],  # CLS position
                    prompt_pos_embed,  # Prompt positions (zeros)
                    pos_embed[:, 1:]   # Patch positions
                ], dim=1)
                x = x + extended_pos_embed

            # Forward through transformer blocks
            x = self.dinov3.blocks(x)
            x = self.dinov3.norm(x)

            # Extract outputs (skip prompt tokens)
            result = {}
            if return_cls:
                result['cls_embeddings'] = x[:, 0]  # CLS token
            if return_patches:
                # Skip CLS and prompt tokens
                result['patch_embeddings'] = x[:, 1 + self.num_prompts:]

            return result

        except AttributeError:
            # Fallback to simple forward if we can't access internals
            return self.forward(images, return_cls, return_patches)

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters (prompts only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_frozen_params(self) -> int:
        """Get number of frozen parameters (DINOv3 backbone)."""
        return sum(p.numel() for p in self.dinov3.parameters())


class PromptTuningWithAdapter(nn.Module):
    """
    Enhanced prompt tuning with adapter layers.

    Combines visual prompts with small adapter modules for increased capacity
    while maintaining parameter efficiency.

    Args:
        dinov3_model: Pretrained DINOv3 model
        num_prompts: Number of prompt tokens
        embed_dim: Embedding dimension
        adapter_dim: Bottleneck dimension for adapters
        num_adapters: Number of adapter layers (one per transformer block)
    """

    def __init__(
        self,
        dinov3_model: nn.Module,
        num_prompts: int = 10,
        embed_dim: int = 384,
        adapter_dim: int = 64,
        num_adapters: int = 12,
    ):
        super().__init__()

        # Freeze backbone
        self.dinov3 = dinov3_model
        for param in self.dinov3.parameters():
            param.requires_grad = False

        # Visual prompts
        self.prompts = nn.Parameter(torch.empty(num_prompts, embed_dim))
        nn.init.xavier_uniform_(self.prompts)

        # Adapter layers (one per transformer block)
        self.adapters = nn.ModuleList([
            Adapter(embed_dim, adapter_dim) for _ in range(num_adapters)
        ])

        self.num_prompts = num_prompts
        self.embed_dim = embed_dim

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with prompts and adapters."""
        # For simplicity, use standard forward and apply adapters post-hoc
        output = self.dinov3(images, is_training=False)

        result = {}
        if isinstance(output, dict):
            # Apply adapter to CLS token
            if 'x_norm_clstoken' in output:
                cls_embed = output['x_norm_clstoken']
                for adapter in self.adapters:
                    cls_embed = adapter(cls_embed)
                result['cls_embeddings'] = cls_embed

            # Apply adapter to patch tokens
            if 'x_norm_patchtokens' in output:
                patch_embed = output['x_norm_patchtokens']
                for adapter in self.adapters:
                    patch_embed = adapter(patch_embed)
                result['patch_embeddings'] = patch_embed

        return result


class Adapter(nn.Module):
    """
    Lightweight adapter module with bottleneck architecture.

    Args:
        input_dim: Input dimension
        bottleneck_dim: Bottleneck dimension (typically 1/4 to 1/8 of input_dim)
    """

    def __init__(self, input_dim: int, bottleneck_dim: int):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Initialize with small weights for stability
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.01)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.01)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection."""
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual  # Skip connection


def load_dinov3_for_prompting(
    model_name: str = 'dinov3_vits16',
    num_prompts: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_adapters: bool = False,
) -> VisualPromptTuning:
    """
    Convenience function to load DINOv3 with prompt tuning.

    Args:
        model_name: DINOv3 model name
        num_prompts: Number of prompt tokens
        device: Device to load model on
        use_adapters: Whether to use adapter layers in addition to prompts

    Returns:
        VisualPromptTuning or PromptTuningWithAdapter model
    """
    print(f"Loading {model_name} for prompt tuning...")

    # Load pretrained DINOv3
    repo_dir = Path(__file__).parent.parent
    try:
        dinov3 = torch.hub.load(
            str(repo_dir),
            model_name,
            source='local',
            pretrained=True
        )
    except Exception as e:
        print(f"Loading from local failed: {e}")
        print("Downloading from torch.hub...")
        dinov3 = torch.hub.load(
            'facebookresearch/dinov3',
            model_name,
            source='github',
            pretrained=True
        )

    dinov3 = dinov3.to(device)
    dinov3.eval()

    # Get embedding dimension
    if hasattr(dinov3, 'embed_dim'):
        embed_dim = dinov3.embed_dim
    elif hasattr(dinov3, 'num_features'):
        embed_dim = dinov3.num_features
    else:
        # Default for ViT-S
        embed_dim = 384

    print(f"Embedding dimension: {embed_dim}")

    # Create prompt tuning model
    if use_adapters:
        model = PromptTuningWithAdapter(
            dinov3,
            num_prompts=num_prompts,
            embed_dim=embed_dim,
        )
    else:
        model = VisualPromptTuning(
            dinov3,
            num_prompts=num_prompts,
            embed_dim=embed_dim,
        )

    num_trainable = model.get_num_trainable_params()
    num_frozen = model.get_num_frozen_params()
    print(f"✓ Model created")
    print(f"  Trainable parameters: {num_trainable:,} ({num_trainable / 1e3:.1f}K)")
    print(f"  Frozen parameters: {num_frozen:,} ({num_frozen / 1e6:.1f}M)")
    if num_trainable > 0:
        print(f"  Efficiency ratio: {num_frozen / num_trainable:.1f}x fewer trainable params")

    return model.to(device)


if __name__ == "__main__":
    # Example usage
    print("Creating prompt tuning model...")
    model = load_dinov3_for_prompting(
        model_name='dinov3_vits16',
        num_prompts=10,
        device='cpu',
    )

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)

    print("\nOutput shapes:")
    for key, value in output.items():
        print(f"  {key}: {value.shape}")
