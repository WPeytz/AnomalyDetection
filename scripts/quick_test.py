"""
Quick test to verify setup with immediate feedback.
"""
import sys

print("=" * 70)
print("Quick Setup Test")
print("=" * 70)

print("\n1. Testing imports...")
try:
    import torch
    import torchvision
    from transformers import AutoImageProcessor, AutoModel
    import numpy as np
    from PIL import Image
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

print("\n2. Checking dataset...")
from pathlib import Path
from mvtec_dataset import MVTecADDataset, get_mvtec_transforms

dataset_path = Path("../mvtec_ad")
if not (dataset_path / "bottle" / "train").exists():
    print(f"✗ Dataset not found at {dataset_path}")
    sys.exit(1)

print("✓ Dataset found")

print("\n3. Loading dataset (bottle category)...")
transform = get_mvtec_transforms(224)
train_dataset = MVTecADDataset(
    root=str(dataset_path),
    category="bottle",
    split="train",
    transform=transform,
)
print(f"✓ Train dataset loaded: {len(train_dataset)} samples")

test_dataset = MVTecADDataset(
    root=str(dataset_path),
    category="bottle",
    split="test",
    transform=transform,
)
print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
print(f"  - Normal: {len(test_dataset.get_normal_samples())}")
print(f"  - Anomaly: {len(test_dataset.get_anomaly_samples())}")

print("\n4. Loading DINOv3 model (this may take 1-2 minutes)...")
sys.stdout.flush()

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
print(f"  Downloading from HuggingFace: {model_name}")
sys.stdout.flush()

processor = AutoImageProcessor.from_pretrained(model_name)
print("✓ Processor loaded")
sys.stdout.flush()

model = AutoModel.from_pretrained(model_name)
print("✓ Model loaded")
print(f"  Hidden size: {model.config.hidden_size}")
print(f"  Number of layers: {model.config.num_hidden_layers}")

print("\n5. Testing inference on one sample...")
sample = train_dataset[0]
image = sample['image'].unsqueeze(0)

with torch.no_grad():
    outputs = model(pixel_values=image)
    embeddings = outputs.last_hidden_state

print(f"✓ Inference successful!")
print(f"  Output shape: {embeddings.shape}")
print(f"  CLS token shape: {outputs.pooler_output.shape if hasattr(outputs, 'pooler_output') else 'N/A'}")

print("\n" + "=" * 70)
print("✓ All tests passed! Setup is working correctly.")
print("=" * 70)

print("\nYou can now run the full anomaly detection:")
print("  python3 anomaly_detection.py --category bottle --root-dir ../mvtec_ad")
