"""
Quick test script to verify the setup is working correctly.

This script tests:
1. Package imports
2. Model loading
3. Dataset loading (if available)
4. Embedding extraction
"""

import sys
import torch


def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('transformers', 'HuggingFace Transformers'),
        ('numpy', 'NumPy'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm'),
        ('PIL', 'Pillow'),
    ]

    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing_packages.append(name)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements_anomaly.txt")
        return False

    print("✓ All packages installed\n")
    return True


def test_device():
    """Test GPU availability."""
    print("Testing device...")

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠️  CUDA not available - will use CPU (slower)")

    print(f"  PyTorch version: {torch.__version__}\n")
    return True


def test_model_loading():
    """Test loading a small DINOv3 model."""
    print("Testing model loading...")

    try:
        from transformers import AutoImageProcessor, AutoModel

        print("  Loading dinov3-vits16 from HuggingFace...")
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        print(f"✓ Model loaded successfully")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Number of layers: {model.config.num_hidden_layers}")
        print(f"  Patch size: {model.config.patch_size}\n")

        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}\n")
        return False


def test_embedding_extraction():
    """Test embedding extraction on a dummy image."""
    print("Testing embedding extraction...")

    try:
        from embedding_extractor import DINOv3EmbeddingExtractor
        import torch

        # Create extractor
        extractor = DINOv3EmbeddingExtractor(
            model_name='dinov3_vits16',
            use_huggingface=True,
        )

        # Create dummy image
        dummy_image = torch.randn(1, 3, 224, 224)

        # Extract embeddings
        embeddings = extractor.extract_patch_embeddings(dummy_image)

        print(f"✓ Embedding extraction successful")
        print(f"  CLS token shape: {embeddings['cls_token'].shape}")
        print(f"  Patch embeddings shape: {embeddings['patch_embeddings'].shape}")
        print(f"  Embedding dimension: {extractor.get_embedding_dim()}\n")

        return True

    except Exception as e:
        print(f"✗ Failed to extract embeddings: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test MVTec AD dataset loading (if available)."""
    print("Testing dataset loading...")

    try:
        from mvtec_dataset import MVTecADDataset, get_mvtec_transforms
        from pathlib import Path

        # Check if dataset exists
        dataset_path = Path("../mvtec_ad")
        if not dataset_path.exists():
            print("⚠️  MVTec AD dataset not found at ../mvtec_ad")
            print("   Run: python download_mvtec.py --root-dir ../mvtec_ad\n")
            return True

        # Try to load a dataset
        transform = get_mvtec_transforms(224)
        dataset = MVTecADDataset(
            root=str(dataset_path),
            category="bottle",
            split="train",
            transform=transform,
        )

        print(f"✓ Dataset loaded successfully")
        print(f"  Category: {dataset.category}")
        print(f"  Split: {dataset.split}")
        print(f"  Total samples: {len(dataset)}")

        # Test loading a sample
        sample = dataset[0]
        print(f"  Sample image shape: {sample['image'].shape}")
        print(f"  Sample label: {sample['label']}")
        print(f"  Sample defect type: {sample['defect_type']}\n")

        return True

    except Exception as e:
        print(f"⚠️  Could not load dataset: {e}")
        print("   This is OK if you haven't downloaded the dataset yet.\n")
        return True


def main():
    """Run all tests."""
    print("="*70)
    print("DINOv3 Anomaly Detection Setup Test")
    print("="*70 + "\n")

    tests = [
        ("Package Imports", test_imports),
        ("Device", test_device),
        ("Model Loading", test_model_loading),
        ("Embedding Extraction", test_embedding_extraction),
        ("Dataset Loading", test_dataset_loading),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}\n")
            results.append((test_name, False))

    # Print summary
    print("="*70)
    print("Test Summary")
    print("="*70)

    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<30} {status}")

    print("="*70 + "\n")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("✓ All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("  1. Download MVTec AD dataset:")
        print("     python download_mvtec.py --root-dir ../mvtec_ad")
        print("\n  2. Run anomaly detection:")
        print("     python anomaly_detection.py --category bottle")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
