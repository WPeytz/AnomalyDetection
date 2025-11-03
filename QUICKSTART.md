# Quick Start Guide: DINOv3 for Anomaly Detection

This guide will help you get started with using DINOv3 for anomaly detection on the MVTec AD dataset.

## Prerequisites

- Python 3.8+
- PyTorch 2.0+
- 8GB+ RAM (16GB+ recommended)
- GPU recommended (but not required)

## Installation

### Step 1: Install Dependencies

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
cd scripts
pip install -r requirements_anomaly.txt
```

### Step 2: Download MVTec AD Dataset

Download a single category for quick testing:

```bash
cd scripts
python3 download_mvtec.py --root-dir ../mvtec_ad
```

This will download all 15 categories (~4.6 GB). The script shows progress for each category.

Alternatively, you can download manually from:
https://www.mvtec.com/company/research/datasets/mvtec-ad

## Running Your First Anomaly Detection

### Example 1: Single Category (Bottle)

```bash
cd scripts
python3 anomaly_detection.py \
    --category bottle \
    --root-dir ../mvtec_ad \
    --model-name dinov3_vits16 \
    --output-dir ../results
```

**What happens:**
1. Downloads DINOv3 model from HuggingFace (~80MB for vits16)
2. Loads training data (normal samples)
3. Extracts embeddings for all training samples
4. Loads test data (normal + anomalous samples)
5. Extracts embeddings for test samples
6. Computes anomaly scores based on similarity to normal samples
7. Evaluates performance (AUROC, Average Precision)
8. Saves results and visualizations

**Expected output:**
```
==========================================
Results for bottle:
==========================================
  AUROC: 0.9500
  Average Precision: 0.9200
==========================================
```

### Example 2: Run on All Categories

```bash
python3 anomaly_detection.py \
    --category all \
    --root-dir ../mvtec_ad \
    --model-name dinov3_vitb16 \
    --output-dir ../results
```

This will process all 15 MVTec AD categories and provide a summary.

## Understanding the Results

After running, you'll find:

```
results/
└── bottle/
    ├── metrics.json              # Performance metrics
    ├── anomaly_scores.npz        # Raw anomaly scores
    └── visualizations/           # Sample visualizations
        ├── top_anomaly_1_score_0.850.png
        ├── top_anomaly_2_score_0.820.png
        ├── ...
        ├── top_normal_1_score_0.120.png
        └── top_normal_2_score_0.150.png
```

**Metrics:**
- **AUROC (Area Under ROC Curve)**: Measures how well the model ranks anomalies vs normal samples. Higher is better (0-1 scale).
- **Average Precision**: Precision-recall based metric. Higher is better (0-1 scale).

## Model Selection

Choose a model based on your needs:

| Model             | Size | Speed | Performance | GPU RAM |
|-------------------|------|-------|-------------|---------|
| dinov3_vits16     | 80MB | Fast  | Good        | ~2GB    |
| dinov3_vitb16     | 350MB| Medium| Better      | ~4GB    |
| dinov3_vitl16     | 1.2GB| Slow  | Best        | ~8GB    |

**Recommendation:** Start with `dinov3_vits16` for quick testing, then try `dinov3_vitb16` for better results.

## Common Options

### Use CLS Token Instead of Patches

```bash
python3 anomaly_detection.py \
    --category bottle \
    --use-cls
```

**When to use:**
- Faster inference
- Less memory
- Good for global defects

**When NOT to use:**
- Small localized defects (patches work better)

### Adjust k-Nearest Neighbors

```bash
python3 anomaly_detection.py \
    --category bottle \
    --k-neighbors 5
```

**Effect:**
- k=1: More sensitive to outliers
- k=5-10: More robust, but may miss subtle anomalies

### Change Image Size

```bash
python3 anomaly_detection.py \
    --category bottle \
    --image-size 448
```

**Trade-offs:**
- Larger size: Better feature resolution, more memory, slower
- Smaller size: Faster, less memory, may miss small defects

## Programmatic Usage

For custom pipelines, you can use the modules directly:

```python
from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
from embedding_extractor import DINOv3EmbeddingExtractor, compute_anomaly_scores
from torch.utils.data import DataLoader

# Load model
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

# Extract embeddings
train_loader = DataLoader(train_dataset, batch_size=32)
train_embeddings = extractor.extract_embeddings_batch(train_loader)

# Do the same for test data...
# Then compute anomaly scores
anomaly_scores = compute_anomaly_scores(
    test_embeddings['patch_embeddings'],
    train_embeddings['patch_embeddings'],
    metric='cosine',
    k=1,
)
```

## Troubleshooting

### Issue: Out of Memory

**Solutions:**
- Reduce batch size: `--batch-size 16`
- Use smaller model: `--model-name dinov3_vits16`
- Reduce image size: `--image-size 224`
- Use CLS token: `--use-cls`

### Issue: Slow Inference

**Solutions:**
- Use GPU if available
- Increase batch size: `--batch-size 64`
- Use smaller model
- Reduce image size

### Issue: Poor Performance

**Solutions:**
- Try patch embeddings (default) instead of CLS token
- Use larger model: `--model-name dinov3_vitb16`
- Adjust k-neighbors: `--k-neighbors 5`
- Increase image size: `--image-size 448`

### Issue: Model Download Fails

**Solutions:**
- Check internet connection
- Try again (downloads resume automatically)
- Download manually from HuggingFace and specify local path

## Next Steps

### 1. Visualize Attention Maps

DINOv3 has strong attention mechanisms. You can visualize which parts of the image the model focuses on.

### 2. Few-Shot Learning

Fine-tune a small adapter on top of DINOv3 features using a few labeled examples.

### 3. Localization

Use patch-level anomaly scores to create heatmaps showing exactly where defects are.

### 4. Combine with Other Models

- Use SAM for segmentation-aware anomaly detection
- Combine multiple model sizes for ensemble

## Resources

- **Scripts Documentation**: See `scripts/README.md` for detailed documentation
- **DINOv3 Paper**: https://arxiv.org/abs/2508.10104
- **MVTec AD Dataset**: https://www.mvtec.com/company/research/datasets/mvtec-ad
- **HuggingFace Models**: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009

## Support

For issues or questions:
1. Check the detailed README in `scripts/README.md`
2. Review the troubleshooting section above
3. Check that all dependencies are installed correctly
