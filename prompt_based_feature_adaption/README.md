# Prompt-Based Feature Adaptation for DINOv3 Anomaly Detection

This module implements visual prompt tuning for adapting pretrained DINOv3 models to anomaly detection tasks without fine-tuning the backbone.

## Results

| Category | Zero-shot | Prompt (full) | Improvement |
|----------|-----------|---------------|-------------|
| Bottle   | 0.7937    | **0.992**     | +19.8%      |
| Carpet   | 0.6200    | **0.960**     | +34.0%      |
| Cable    | 0.6089    | **0.932**     | +32.3%      |
| Screw    | 0.5325    | **0.773**     | +24.1%      |
| Hazelnut | 0.5204    | **0.928**     | +40.8%      |
| **Average** | **0.6151** | **0.917** | **+30.2%** |

## Key Insights

1. **Per-patch modulation is essential**: Global transforms preserve relative cosine similarities and yield no improvement. Each patch must attend to prompts independently.

2. **Full defect samples needed**: Training with only 5 samples (few-shot) provides negligible improvement. Using all available defect samples enables the model to learn generalizable feature adaptations.

3. **Fast training**: ~90 seconds for 100 epochs on M1 Mac.

## Quick Start

### Train Prompts (Full)

```bash
python prompt_based_feature_adaption/run_prompt_full.py \
    --categories bottle carpet cable screw hazelnut \
    --num-epochs 100
```

### Train Single Category

```bash
python prompt_based_feature_adaption/train_prompts.py \
    --category carpet \
    --num-prompts 10 \
    --num-defect-samples 89 \
    --num-epochs 100 \
    --modulation-type per_patch
```

### Evaluate

```bash
python prompt_based_feature_adaption/evaluate_prompts.py \
    --category carpet \
    --checkpoint prompt_based_feature_adaption/checkpoints/prompt_full/carpet_prompts.pt
```

### Generate Anomaly Map

```bash
python prompt_based_feature_adaption/generate_single_image_anomaly_map.py \
    --image-path mvtec_ad/carpet/test/thread/008.png \
    --checkpoint prompt_based_feature_adaption/checkpoints/prompt_full/carpet_prompts.pt \
    --category carpet
```

## Architecture

```
Input Image (224x224)
    |
DINOv3 Feature Extraction (FROZEN)
    |
Patch Embeddings [B, 196, 384]
    |
Per-Patch Prompt Attention
    - Each patch attends to learnable prompts
    - Weighted combination produces per-patch modulation
    |
Residual Connection
    - output = patch_embed + alpha * prompt_effect
    |
Modulated Embeddings for Anomaly Scoring
```

## Files

```
prompt_based_feature_adaption/
├── prompt_model.py              # VisualPromptTuning model
├── train_prompts.py             # Training script (single category)
├── evaluate_prompts.py          # Evaluation script
├── run_prompt_full.py           # Run experiments across all categories
├── generate_single_image_anomaly_map.py  # Visualize anomaly maps
├── checkpoints/
│   └── prompt_full/             # Checkpoints using all defect samples
├── results/
│   └── prompt_full/             # Evaluation results
└── README.md
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--category` | required | MVTec AD category |
| `--num-prompts` | 10 | Number of learnable prompt tokens |
| `--num-defect-samples` | 10 | Defect samples for training (use all for best results) |
| `--num-epochs` | 100 | Training epochs |
| `--modulation-type` | per_patch | `per_patch`, `learned_transform`, or `global` |
| `--scaling-factor` | 0.1 | Residual scaling factor |

## Why Global Modulation Fails

Our initial approach applied identical offsets to all patches:
```
patch_embed = patch_embed + alpha * global_prompt_effect
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
