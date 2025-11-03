# DINOv3 for Anomaly Detection on MVTec AD

This project demonstrates how to use pretrained DINOv3 vision foundation models for zero-shot and few-shot anomaly detection on the MVTec Anomaly Detection (AD) dataset.

## Overview

The pipeline extracts dense feature embeddings from DINOv3 and uses similarity-based methods to detect anomalies without any task-specific training. This approach leverages:

- **Pretrained DINOv3 models**: Using either patch-level or CLS token embeddings
- **Zero-shot detection**: No training on MVTec AD required
- **Similarity metrics**: Cosine similarity or Euclidean distance to normal samples
- **K-nearest neighbors**: Anomaly scoring based on similarity to k-nearest normal samples

## Project Structure

```
scripts/
├── download_mvtec.py       # Download MVTec AD dataset
├── mvtec_dataset.py         # PyTorch dataset loader for MVTec AD
├── embedding_extractor.py   # DINOv3 embedding extraction utilities
├── anomaly_detection.py     # Main anomaly detection pipeline
└── README.md                # This file

mvtec_ad/                    # MVTec AD dataset (downloaded)
├── bottle/
├── cable/
├── ...
└── zipper/

results/                     # Output directory
├── bottle/
│   ├── metrics.json
│   ├── anomaly_scores.npz
│   └── visualizations/
└── ...
```

## Setup

### 1. Install Dependencies

```bash
# Install PyTorch (with CUDA if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install transformers
pip install scikit-learn
pip install matplotlib
pip install tqdm
pip install Pillow
```

Or if using the conda environment from the main repo:

```bash
micromamba env create -f ../conda.yaml
micromamba activate dinov3
pip install transformers  # Add HuggingFace transformers
```

### 2. Download MVTec AD Dataset

```bash
python download_mvtec.py --root-dir ../mvtec_ad
```

This will download all 15 categories (~4.6 GB total) to the `mvtec_ad` directory.

To download only specific categories, modify the script or download manually from:
https://www.mvtec.com/company/research/datasets/mvtec-ad

## Usage

### Quick Start: Single Category

Run anomaly detection on a single category (e.g., bottle):

```bash
python anomaly_detection.py \
    --category bottle \
    --root-dir ../mvtec_ad \
    --model-name dinov3_vits16 \
    --output-dir ../results
```

### Run on All Categories

```bash
python anomaly_detection.py \
    --category all \
    --root-dir ../mvtec_ad \
    --model-name dinov3_vitb16 \
    --output-dir ../results
```

### Options

```
--category            Category to evaluate (or 'all')
                      Options: bottle, cable, capsule, carpet, grid,
                               hazelnut, leather, metal_nut, pill, screw,
                               tile, toothbrush, transistor, wood, zipper, all

--model-name          DINOv3 model to use
                      Options: dinov3_vits16 (default)
                               dinov3_vitb16
                               dinov3_vitl16
                               dinov3_vith16plus
                               dinov3_vit7b16

--batch-size          Batch size for processing (default: 32)

--image-size          Input image size (default: 224)

--use-cls             Use CLS token instead of patch embeddings

--k-neighbors         Number of nearest neighbors for scoring (default: 1)

--visualize-samples   Number of samples to visualize (default: 5)

--output-dir          Directory to save results (default: ./results)
```

### Examples

**Use larger model for better performance:**
```bash
python anomaly_detection.py \
    --category bottle \
    --model-name dinov3_vitb16
```

**Use CLS token embeddings instead of patches:**
```bash
python anomaly_detection.py \
    --category bottle \
    --use-cls
```

**Use k=5 nearest neighbors:**
```bash
python anomaly_detection.py \
    --category bottle \
    --k-neighbors 5
```

## Understanding the Pipeline

### 1. Embedding Extraction

The pipeline extracts two types of embeddings from DINOv3:

- **CLS Token Embedding**: Global image representation (1 × D)
- **Patch Embeddings**: Local patch-level features (P × D, where P = number of patches)

For a 224×224 image with patch size 16×16, you get 196 patch embeddings.

### 2. Anomaly Scoring

For each test image:
1. Extract embeddings using pretrained DINOv3
2. Compute similarity to all normal training samples
3. Find k-nearest neighbors (highest similarity)
4. Anomaly score = 1 - (average similarity to k-nearest neighbors)

Higher scores indicate more anomalous samples.

### 3. Evaluation Metrics

- **AUROC (Area Under ROC Curve)**: Measures ranking quality of anomaly scores
- **Average Precision (AP)**: Precision-recall based metric

## Results

Expected performance with `dinov3_vits16`:

| Category    | AUROC | Average Precision |
|-------------|-------|-------------------|
| bottle      | ~0.95 | ~0.92            |
| cable       | ~0.85 | ~0.80            |
| capsule     | ~0.90 | ~0.85            |
| ...         | ...   | ...              |

Results will vary based on:
- Model size (larger models generally perform better)
- Using patches vs CLS token (patches typically better for localized defects)
- Value of k (k=1 usually works well)

## Advanced Usage

### Custom Embedding Extraction

```python
from embedding_extractor import DINOv3EmbeddingExtractor
from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
from torch.utils.data import DataLoader

# Initialize extractor
extractor = DINOv3EmbeddingExtractor(
    model_name='dinov3_vits16',
    use_huggingface=True,
)

# Load data
dataset = MVTecADDataset(
    root="./mvtec_ad",
    category="bottle",
    split="train",
    transform=get_mvtec_transforms(224),
)
loader = DataLoader(dataset, batch_size=32)

# Extract embeddings
embeddings = extractor.extract_embeddings_batch(loader)
print(embeddings['cls_embeddings'].shape)     # (N, D)
print(embeddings['patch_embeddings'].shape)   # (N, P, D)
```

### Custom Anomaly Detection

```python
from embedding_extractor import compute_anomaly_scores

# Compute anomaly scores
anomaly_scores = compute_anomaly_scores(
    test_embeddings,
    normal_embeddings,
    metric='cosine',  # or 'euclidean'
    k=1,
)
```

## Extensions and Next Steps

### 1. Fine-tuning

Fine-tune DINOv3 on MVTec AD categories for improved performance:
- Add a small adapter head on top of frozen DINOv3 features
- Train with few-shot learning approaches

### 2. Better Anomaly Scoring

- **PatchCore**: Memory bank of normal patch features with coreset selection
- **Mahalanobis distance**: Use multivariate Gaussian to model normal distribution
- **Ensemble**: Combine multiple layers or models

### 3. Visualization

- Generate attention maps using DINOv3 attention weights
- Create heatmaps highlighting anomalous regions
- Compare with ground truth masks for pixel-level evaluation

### 4. Other Foundation Models

- **SAM (Segment Anything Model)**: Combine with DINOv3 for segmentation-aware anomaly detection
- **Mirroring DINO**: Use both left-right consistency and embeddings

## Troubleshooting

**Out of memory errors:**
- Reduce `--batch-size`
- Use smaller model (e.g., `dinov3_vits16` instead of `dinov3_vitl16`)
- Reduce `--image-size`

**Slow inference:**
- Ensure CUDA is available: `torch.cuda.is_available()`
- Use smaller model for faster inference
- Increase batch size if memory allows

**Poor performance:**
- Try using patch embeddings instead of CLS token (remove `--use-cls`)
- Try larger models (`dinov3_vitb16` or `dinov3_vitl16`)
- Adjust k-neighbors parameter

## References

- DINOv3 Paper: https://arxiv.org/abs/2508.10104
- MVTec AD Dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
- HuggingFace DINOv3: https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009

## License

This code is released under the DINOv3 License (see main repo).
MVTec AD dataset has its own license - see dataset website.
