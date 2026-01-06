import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from transformers import AutoTokenizer
from sklearn.model_selection import KFold
import numpy as np
import os
from tqdm import tqdm

# Import implementation
from dataset import DTIDataset, collate_dti
from architectures import get_model
from train import validate

def evaluate_folds(data_path, model_name='mamba_bilstm', folds=5, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data (Same as train.py)
    print("Loading Tokenizers...")
    smi_tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    prot_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    
    print("Initializing Dataset (this may take a minute)...")
    dataset = DTIDataset(data_path, smi_tokenizer, prot_tokenizer, max_len_drug=128, max_len_prot=350)
    
    # 2. Re-create Splits (Must match train.py random_state=42)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    accuracies = []
    losses = []
    
    print(f"\nEvaluating 5-Fold Performance for {model_name}...")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        fold_idx = fold + 1
        print(f"\n--- Evaluating Fold {fold_idx}/{folds} ---")
        
        # Path to best model
        model_path = os.path.join('checkpoints', f'model_fold_{fold_idx}_best.pth')
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Skipping.")
            continue
            
        # Create Validation Loader
        val_subsampler = SubsetRandomSampler(val_ids)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, collate_fn=collate_dti)
        
        # Load Model
        model = get_model(model_name, drug_dim=256, prot_dim=512, hidden_dim=256).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Validate
        criterion = nn.BCELoss()
        result = validate(model, val_loader, criterion, device)
        
        print(f"Fold {fold_idx} Best Acc: {result['acc']:.2f}% (Loss: {result['loss']:.4f})")
        accuracies.append(result['acc'])
        losses.append(result['loss'])
        
    print("\n" + "="*30)
    print(f"Final Results:")
    print(f"Average Accuracy: {np.mean(accuracies):.2f}%")
    print(f"Average Loss:     {np.mean(losses):.4f}")
    print("="*30)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r'e:\基于Mamba与BiLSTM混合架构\data\Davis.txt')
    parser.add_argument('--model_name', type=str, default='mamba_bilstm')
    args = parser.parse_args()
    
    evaluate_folds(args.data, model_name=args.model_name)
