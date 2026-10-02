import sys
import numpy as np
import cv2
import matplotlib
import imagecodecs
import torch
import torchvision
import torch_geometric
import compressai


print("=" * 60)
print("multimedia312 ENVIRONMENT TEST")
print("=" * 60)

print(f"Python       : {sys.version.split()[0]}")
print(f"Python path  : {sys.executable}")

print("-" * 60)

print(f"NumPy        : {np.__version__}")
print(f"OpenCV       : {cv2.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"ImageCodecs  : {imagecodecs.__version__}")
print(f"PyTorch      : {torch.__version__}")
print(f"TorchVision  : {torchvision.__version__}")
print(f"PyG          : {torch_geometric.__version__}")
print(f"CompressAI   : {compressai.__version__}")

print("-" * 60)

print(f"CUDA build   : {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")
    print(f"GPU count    : {torch.cuda.device_count()}")

    # Simple GPU computation
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = x * 2

    print(f"GPU test     : {y}")
else:
    print("GPU test     : FAILED - CUDA unavailable")

print("-" * 60)

# Simple NumPy test
a = np.array([1, 2, 3])
print(f"NumPy test   : {a * 2}")

# Simple OpenCV test
img = np.zeros((100, 100, 3), dtype=np.uint8)
print(f"OpenCV test  : image created {img.shape}")

print("=" * 60)
print("ALL TESTS COMPLETED")
print("=" * 60)