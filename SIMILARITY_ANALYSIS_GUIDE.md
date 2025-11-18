# Similarity Analysis: Normal vs Defective Samples

## Overview

This guide explains how to use the similarity analysis tool to compare normal and defective samples using DINOv3 embeddings.

## What It Does

The `similarity_analysis.py` script performs comprehensive similarity metric analysis between normal and defective samples:

### 1. **Intra-class Similarity Analysis**
- **Normal-to-Normal**: How similar are normal samples to each other?
- **Defect-to-Defect**: How similar are defective samples to each other?

### 2. **Inter-class Similarity Analysis**
- **Normal-to-Defect**: How similar are normal samples to defective samples?

### 3. **Statistical Testing**
- **T-test**: Tests if normal-normal and normal-defect similarities are significantly different
- **Cohen's d**: Measures effect size (how large is the difference?)
- **Separability**: Quantifies how well the classes can be separated

### 4. **Visualizations**
- **Distribution plots**: Histograms, box plots, and violin plots
- **Embedding space visualization**: t-SNE and PCA projections showing cluster separation
- **Statistical summary table**: All metrics in one view

## Usage

### Basic Usage

```bash
cd scripts
python3 similarity_analysis.py --category bottle --root-dir ../mvtec_ad
```

### Options

```bash
python3 similarity_analysis.py \
    --category bottle \              # MVTec category to analyze
    --root-dir ../mvtec_ad \         # Dataset root directory
    --model-name dinov3_vits16 \     # DINOv3 model variant
    --metric cosine \                # Similarity metric (cosine/euclidean)
    --use-cls \                      # Use CLS token instead of patches
    --output-dir ../results          # Output directory
```

### Run on Multiple Categories

```bash
# Analyze bottle
python3 similarity_analysis.py --category bottle

# Analyze cable
python3 similarity_analysis.py --category cable

# Analyze all categories
for category in bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper; do
    python3 similarity_analysis.py --category $category
done
```

## Results Interpretation

### Example: Bottle Category

```
Normal-to-Normal Similarity:
  Mean: 0.9821 ± 0.0065
  Range: [0.9638, 0.9950]

Defect-to-Defect Similarity:
  Mean: 0.8832 ± 0.0617
  Range: [0.6726, 0.9830]

Normal-to-Defect Similarity:
  Mean: 0.9048 ± 0.0503
  Range: [0.7406, 0.9885]

Statistical Significance:
  T-test p-value: 1.36e-156
  Significant: True
  Cohen's d: 2.1534
  Separability: 1.3594
```

**Interpretation:**
- ✅ **Normal samples are very similar to each other** (0.9821 mean similarity)
- ✅ **Clear separation** between normal and defect samples (p < 0.001)
- ✅ **Large effect size** (Cohen's d = 2.15, considered "very large")
- ✅ **High separability** (1.36) indicates anomaly detection should work well

### Example: Cable Category

```
Normal-to-Normal Similarity:
  Mean: 0.8997 ± 0.0318
  Range: [0.7725, 0.9722]

Defect-to-Defect Similarity:
  Mean: 0.8082 ± 0.1059
  Range: [0.1933, 0.9556]

Normal-to-Defect Similarity:
  Mean: 0.8428 ± 0.0902
  Range: [0.2196, 0.9527]

Statistical Significance:
  T-test p-value: 1.85e-250
  Significant: True
  Cohen's d: 0.8415
  Separability: 0.4665
```

**Interpretation:**
- ✅ **Still significant separation** (p < 0.001)
- ⚠️ **Medium effect size** (Cohen's d = 0.84)
- ⚠️ **Lower separability** (0.47) - more challenging than bottle
- 💡 **Higher defect variability** (std = 0.1059) suggests diverse defect types

## Understanding the Metrics

### Similarity Scores
- **Range**: [0, 1] for cosine similarity (1 = identical, 0 = orthogonal)
- **Higher is better** for intra-class similarity
- **Lower is better** for inter-class separation

### Cohen's d (Effect Size)
- **< 0.2**: Small effect
- **0.2 - 0.5**: Small to medium effect
- **0.5 - 0.8**: Medium to large effect
- **> 0.8**: Large effect
- **> 1.2**: Very large effect

### Separability Score
- **> 1.0**: Excellent separation
- **0.5 - 1.0**: Good separation
- **0.2 - 0.5**: Moderate separation
- **< 0.2**: Poor separation

## Output Files

After running the analysis, you'll find:

```
results/
└── {category}/
    └── similarity_analysis/
        ├── similarity_metrics.json          # Numerical results
        ├── similarity_distributions.png     # Distribution plots
        ├── embedding_space_tsne.png        # t-SNE visualization
        └── embedding_space_pca.png         # PCA visualization
```

### Generated Visualizations

1. **similarity_distributions.png**:
   - Histogram showing overlap/separation of distributions
   - Box plots comparing the three similarity types
   - Violin plots showing distribution shapes
   - Statistical summary table

2. **embedding_space_tsne.png**:
   - 2D t-SNE projection of embeddings
   - Green dots = normal samples
   - Red dots = defect samples
   - Shows clustering and separation visually

3. **embedding_space_pca.png**:
   - 2D PCA projection (linear dimensionality reduction)
   - Same color coding as t-SNE
   - Shows linear separability

## Key Findings

### Bottle Category (Easy)
- **Excellent separability** (1.36)
- Normal samples form tight cluster
- Clear boundary between normal and defects
- **Perfect AUROC** (1.0) expected and achieved

### Cable Category (Moderate)
- **Good but lower separability** (0.47)
- More overlap between normal and defect distributions
- Higher variance in defect samples (multiple defect types)
- **Good AUROC** (0.94) as expected

## Next Steps

### For Your Project

1. **Week 2 Analysis**:
   - ✅ Run similarity analysis on all 15 MVTec categories
   - ✅ Compare which categories are easy/hard for zero-shot detection
   - ✅ Use these metrics to explain AUROC performance differences

2. **Report Insights**:
   - Categories with high separability → Zero-shot works well
   - Categories with low separability → May need few-shot or fine-tuning
   - Use Cohen's d to quantify difficulty objectively

3. **Future Improvements**:
   - Try different similarity metrics (euclidean vs cosine)
   - Compare CLS token vs patch embeddings
   - Analyze per-defect-type similarities

## Example Usage in Report

> "Our similarity analysis reveals significant inter-class separation for the bottle
> category (Cohen's d = 2.15, p < 0.001), with normal samples exhibiting high
> intra-class similarity (mean = 0.982 ± 0.007) compared to normal-to-defect
> similarity (mean = 0.905 ± 0.050). This large effect size explains the perfect
> AUROC (1.0) achieved in zero-shot anomaly detection. In contrast, the cable
> category shows moderate separability (Cohen's d = 0.84), correlating with lower
> but still strong performance (AUROC = 0.94)."

## Troubleshooting

### "Not enough samples" error
- Some categories may have very few test samples
- Try using `--use-cls` for faster computation

### Out of memory
- Reduce batch size: `--batch-size 16`
- Use smaller model: `--model-name dinov3_vits16`

### Slow t-SNE computation
- t-SNE can be slow for large datasets
- PCA is much faster and often sufficient

## References

- **Cohen's d**: https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
- **t-SNE**: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- **Statistical tests**: Welch's t-test for unequal variances
