"""
Prompt-Based Feature Adaptation for DINOv3 Anomaly Detection.

This package provides visual prompt tuning capabilities for adapting
pretrained DINOv3 models to specific anomaly detection tasks.
"""

from .prompt_model import (
    VisualPromptTuning,
    PromptTuningWithAdapter,
    Adapter,
    load_dinov3_for_prompting,
)

__all__ = [
    'VisualPromptTuning',
    'PromptTuningWithAdapter',
    'Adapter',
    'load_dinov3_for_prompting',
]

__version__ = '1.0.0'
