import sys
print(f"Python: {sys.version}")
try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"Torch path: {torch.__file__}")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    import torchvision
    print(f"Torchvision: {torchvision.__version__}")
except ImportError as e:
    print(f"Error importing torchvision: {e}")
