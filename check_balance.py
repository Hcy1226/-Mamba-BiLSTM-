from dataset import DTIDataset
from transformers import AutoTokenizer
import torch
import numpy as np

# Use relative path as per recent fix
data_path = './data/Davis.txt'

print(f"Checking class balance for {data_path}...")

# Simple file read to avoid loading heavy tokenizers if not strictly necessary, 
# but using Dataset class ensures we see exactly what the model sees.
# However, to be fast, let's just parse the file text directly first.

try:
    with open(data_path, 'r') as f:
        lines = f.readlines()
        
    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            try:
                # Label is usually the last element
                l = float(parts[-1])
                labels.append(l)
            except:
                pass

    total = len(labels)
    if total == 0:
        print("No data found.")
        exit()

    ones = sum(labels)
    zeros = total - ones
    
    print(f"Total samples: {total}")
    print(f"Positive (1): {ones} ({ones/total*100:.2f}%)")
    print(f"Negative (0): {zeros} ({zeros/total*100:.2f}%)")

except Exception as e:
    print(f"Error: {e}")
