"""Diagnose import issues."""
import sys

print("Step 1: Import torch")
sys.stdout.flush()
import torch
print("✓ torch imported")
sys.stdout.flush()

print("\nStep 2: Import torchvision")
sys.stdout.flush()
import torchvision
print("✓ torchvision imported")
sys.stdout.flush()

print("\nStep 3: Import numpy")
sys.stdout.flush()
import numpy as np
print("✓ numpy imported")
sys.stdout.flush()

print("\nStep 4: Import PIL")
sys.stdout.flush()
from PIL import Image
print("✓ PIL imported")
sys.stdout.flush()

print("\nStep 5: Import transformers")
sys.stdout.flush()
from transformers import AutoImageProcessor
print("✓ transformers.AutoImageProcessor imported")
sys.stdout.flush()

print("\nStep 6: Import AutoModel")
sys.stdout.flush()
from transformers import AutoModel
print("✓ transformers.AutoModel imported")
sys.stdout.flush()

print("\n✓ All imports successful!")
