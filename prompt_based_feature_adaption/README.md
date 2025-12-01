# Prompt-Based Feature Adaptation for DINOv3 Anomaly Detection

This module implements visual prompt tuning for adapting pretrained DINOv3 models to anomaly detection tasks without fine-tuning the backbone.

## Results

| Category | Baseline | Adapted | Improvement |
|----------|----------|---------|-------------|
| Screw    | 0.764    | **0.984** | +28.9%    |
| Carpet   | 0.958    | **0.996** | +3.9%     |
| Bottle   | 0.992    | **1.000** | +0.8%     |
| Cable    | 0.930    | **0.936** | +0.6%     |
| Hazelnut | 0.928    | 0.901     | -2.8%     |
| **Average** | **0.914** | **0.963** | **+5.4%** |

## Key Insights

1. **Per-patch modulation is essential**: Global transforms preserve relative cosine similarities and yield no improvement. Each patch must attend to prompts independently.

2. **30 defect samples needed**: Training with only 10 samples leads to overfitting. 30 samples provide sufficient variety for generalization.

3. **Fast training**: ~90 seconds for 100 epochs on M1 Mac.

## Quick Start

### Train Prompts

```bash
./venv/bin/python prompt_based_feature_adaption/train_prompts.py \
    --category screw \
    --num-prompts 10 \
    --num-defect-samples 30 \
    --num-epochs 100 \
    --modulation-type learned_transform
```

### Evaluate

```bash
./venv/bin/python prompt_based_feature_adaption/evaluate_prompts.py \
    --category screw \
    --checkpoint prompt_based_feature_adaption/checkpoints/screw_prompts.pt
```

## Architecture

```
Input Image (224×224)
    ↓
DINOv3 Feature Extraction (FROZEN)
    ↓
Patch Embeddings [B, 196, 384]
    ↓
Per-Patch Prompt Attention
    - Each patch attends to learnable prompts
    - Weighted combination produces per-patch modulation
    ↓
Learned MLP Transform
    - Concatenate [patch_embed, prompt_effect]
    - Transform through 2-layer MLP
    ↓
Residual Connection
    - output = patch_embed + α * transformed
    ↓
Modulated Embeddings for Anomaly Scoring
```

## Files

```
prompt_based_feature_adaption/
├── prompt_model.py          # VisualPromptTuning model
├── train_prompts.py         # Training script
├── evaluate_prompts.py      # Evaluation script
├── checkpoints/
│   ├── learned_transform/   # Checkpoints (10 defect samples)
│   └── more_defects/        # Checkpoints (30 defect samples) ← Best results
└── README.md
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--category` | required | MVTec AD category |
| `--num-prompts` | 10 | Number of learnable prompt tokens |
| `--num-defect-samples` | 10 | Defect samples for training (use 30) |
| `--num-epochs` | 100 | Training epochs |
| `--modulation-type` | learned_transform | `per_patch`, `learned_transform`, or `global` |
| `--scaling-factor` | 0.1 | Residual scaling factor |

## Why Global Modulation Fails

Our initial approach applied identical offsets to all patches:
```
patch_embed = patch_embed + α * global_prompt_effect
```

This preserves relative cosine similarities between patches, so anomaly rankings don't change. The per-patch approach breaks this symmetry by allowing each patch to receive a different modulation based on its attention to the prompts.

## Citation

```bibtex
@misc{dinov3_prompt_adaptation,
  title={Prompt-Based Feature Adaptation for DINOv3 Anomaly Detection},
  author={Peytz, William and Gonzenbach, Rian and Wassmer, Alexander},
  year={2025}
}
```
