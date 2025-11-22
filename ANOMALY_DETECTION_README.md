# DINOv3 for MVTec AD Anomaly Detection

This project extends the official [DINOv3](https://github.com/facebookresearch/dinov3) repository with custom scripts for zero-shot and few-shot anomaly detection on the MVTec Anomaly Detection (AD) dataset.

## 🎯 Project Overview

We apply pretrained DINOv3 vision foundation models for industrial anomaly detection without additional training, using patch-level embeddings and similarity-based methods to detect surface defects and irregular textures.

## 📊 Results

**Bottle Category (Zero-Shot):**
- AUROC: 1.0000
- Average Precision: 1.0000

## 🆕 Custom Additions

This repository includes the following custom scripts (located in `scripts/`):

1. **`download_mvtec.py`** - Automated MVTec AD dataset downloader
2. **`mvtec_dataset.py`** - PyTorch Dataset class for MVTec AD
3. **`embedding_extractor.py`** - DINOv3 embedding extraction utilities
4. **`anomaly_detection.py`** - Full anomaly detection pipeline
5. **`simple_anomaly_detection.py`** - Simplified version using torch.hub
6. **`test_setup.py`** - Setup verification script

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment with Python 3.10+
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
cd scripts
python3 download_mvtec.py --root-dir ../mvtec_ad
```

### 3. Download DINOv3 Model

Request access at: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/

Download the model weights and place in `~/.cache/torch/hub/checkpoints/`

### 4. Run Anomaly Detection

```bash
cd scripts
../venv/bin/python3 simple_anomaly_detection.py \
    --category bottle \
    --root-dir ../mvtec_ad \
    --output-dir ../results
```

## 📁 Repository Structure

```
dinov3-main/
├── scripts/                    # Custom anomaly detection scripts
│   ├── download_mvtec.py
│   ├── mvtec_dataset.py
│   ├── embedding_extractor.py
│   ├── anomaly_detection.py
│   ├── simple_anomaly_detection.py
│   └── requirements_anomaly.txt
├── dinov3/                     # Original DINOv3 code
├── mvtec_ad/                   # MVTec AD dataset (download separately)
├── results/                    # Output directory
├── QUICKSTART.md              # Quick start guide
└── LICENSE.md                 # DINOv3 License
```

## 🔬 Methodology

1. **Feature Extraction**: Extract dense patch-level embeddings using pretrained DINOv3
2. **Normal Reference**: Build reference from normal training samples
3. **Anomaly Scoring**: Compute anomaly scores based on cosine similarity to k-nearest normal samples
4. **Evaluation**: Measure performance using AUROC and Average Precision

## 📈 Available Categories

All 15 MVTec AD categories are supported:
- bottle, cable, capsule, carpet, grid
- hazelnut, leather, metal_nut, pill, screw
- tile, toothbrush, transistor, wood, zipper

## 🛠️ Advanced Usage

### Run on All Categories
```bash
../venv/bin/python3 simple_anomaly_detection.py --category all
```

### Use Different Model Sizes
```bash
# Larger models (better performance, slower)
../venv/bin/python3 simple_anomaly_detection.py \
    --category cable \
    --model-name dinov3_vitb16
```

### Experiment with Parameters
```bash
# Use CLS token (faster)
../venv/bin/python3 simple_anomaly_detection.py \
    --category hazelnut \
    --use-cls

# Adjust k-neighbors
../venv/bin/python3 simple_anomaly_detection.py \
    --category carpet \
    --k-neighbors 5
```

## 📝 Citation

If you use this code or methodology in your research, please cite:

**DINOv3:**
```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and others},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104},
}
```

**MVTec AD Dataset:**
```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A comprehensive real-world dataset for unsupervised anomaly detection},
  author={Bergmann, Paul and others},
  booktitle={CVPR},
  year={2019}
}
```

## 📄 License

### Original DINOv3 Code
The DINOv3 model and original codebase are licensed under the [DINOv3 License](LICENSE.md) by Meta Platforms, Inc.

### Custom Scripts
The custom anomaly detection scripts in the `scripts/` directory are provided as derivative works under the same DINOv3 License terms.

### MVTec AD Dataset
The MVTec AD dataset has its own license. See: https://www.mvtec.com/company/research/datasets/mvtec-ad

## 🤝 Acknowledgments

- **DINOv3** by Meta AI Research
- **MVTec AD Dataset** by MVTec Software GmbH
- Custom implementation by [Your Group Name]

## 📧 Contact

For questions about the custom scripts, contact [your email/group].

For questions about DINOv3, see the [official repository](https://github.com/facebookresearch/dinov3).

---

**Note**: This project uses pretrained DINOv3 models which require acceptance of Meta's license terms.
