import sys

print("1. Importing torch...")
import torch
print(f"   Success. Torch version: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

print("2. Importing torch_geometric...")
try:
    import torch_geometric
    print(f"   Success. PyG version: {torch_geometric.__version__}")
except ImportError as e:
    print(f"   Failed: {e}")

print("3. Testing GCNConv (PyG)...")
try:
    from torch_geometric.nn import GCNConv
    conv = GCNConv(16, 32)
    print("   Success. GCNConv initialized.")
except Exception as e:
    print(f"   Failed: {e}")

print("4. Testing GATConv (PyG)...")
try:
    from torch_geometric.nn import GATConv
    gat = GATConv(16, 32, heads=1)
    print("   Success. GATConv initialized.")
except Exception as e:
    print(f"   Failed: {e}")

print("5. Importing Transformers...")
try:
    from transformers import AutoModel, AutoTokenizer
    print("   Success. Transformers imported.")
except Exception as e:
    print(f"   Failed: {e}")

print("6. Testing AutoTokenizer (ChemBERTa)...")
try:
    tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    print("   Success. Tokenizer loaded.")
except Exception as e:
    print(f"   Failed loading Tokenizer: {e}")

print("7. Testing AutoModel (ChemBERTa)...")
try:
    model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    print("   Success. ChemBERTa loaded.")
except Exception as e:
    print(f"   Failed loading ChemBERTa: {e}")

print("Diagnostic Complete.")
