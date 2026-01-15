import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix

# 导入模块
from architectures import get_model
from dataset import DTIDataset, collate_dti

def train_model(data_path, data_name='Davis', batch_size=64, epochs=100, lr=1e-4, folds=5, model_name='mamba_bilstm', fine_tune=False, hidden_dim=512, debug=False):
    """
    训练主函数 (Enhanced for Comparative Experiments)
    Structure: {data_name}/{model_name}/train_result/
    """
    # 1. 构造输出目录
    # e.g., Davis/Mamba-BiLSTM/train_result
    base_output_dir = os.path.join(data_name, model_name, 'train_result')
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        
    print(f"Training Output Directory: {base_output_dir}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 2. 数据准备 ---
    print("Loading Tokenizers (ChemBERTa & ESM-2)...")
    try:
        smi_tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
        prot_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return

    print("Initializing Dataset...")
    dataset = DTIDataset(data_path, smi_tokenizer, prot_tokenizer, max_len_drug=128, max_len_prot=350)
    
    if debug:
        print(f"!!! DEBUG MODE: Using small subset of data ({min(1000, len(dataset))} samples) !!!")
        dataset.slice_subset(1000)
        batch_size = 8
        epochs = 2 
        folds = 5

    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # K-Fold Split
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    # Metrics Storage (Detailed)
    # Storing final epoch metrics for each fold
    all_folds_metrics = []
    
    print(f"Starting {folds}-Fold Cross Validation for Model: {model_name} on {data_name}...")
    
    # Initialize Log File with Headers (Simplified)
    # Fold, ACC, Sn, Sp, Pre, F1, MCC, AUC
    log_path = os.path.join(base_output_dir, 'train_log.csv')
    with open(log_path, 'w') as f:
        f.write("Fold,ACC,Sn,Sp,Pre,F1,MCC,AUC\n")

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{folds} ---")
        
        # Subsets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, collate_fn=collate_dti)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, collate_fn=collate_dti)
        
        # --- 3. 模型初始化 ---
        print(f"Initializing {model_name}...")
        try:
            # Pass vocab sizes for baseline models that use Embedding layers
            model = get_model(model_name, 
                              drug_dim=256, prot_dim=512, hidden_dim=hidden_dim, fine_tune=fine_tune,
                              drug_vocab_size=len(smi_tokenizer),
                              prot_vocab_size=len(prot_tokenizer)).to(device)
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
            return
        
        # --- 4. 训练配置 ---
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr) 
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        
        best_acc = 0.0
        best_metrics = None
        
        # --- 5. 训练循环 ---
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs}", leave=False)
            for batch in loop:
                d_input = (batch['drug_input'][0].to(device), batch['drug_input'][1].to(device), batch['drug_input'][2].to(device))
                p_input = (batch['prot_input'][0].to(device), batch['prot_input'][1].to(device), batch['prot_input'][2].to(device))
                labels = batch['labels'].to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(d_input, p_input) # logits
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            avg_loss = total_loss / len(train_loader)
            
            # --- 验证 (Full Metrics) ---
            val_metrics, val_preds, val_targets, val_probs = validate_full(model, val_loader, criterion, device)
            
            # Step Scheduler
            scheduler.step(val_metrics['ACC'])
            
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_metrics['ACC']:.2f}%, AUC={val_metrics['AUC']:.4f}")
            
            # 记录最佳模型 (Updates per epoch, but only logs final best at end of fold)
            if val_metrics['ACC'] > best_acc:
                best_acc = val_metrics['ACC']
                best_metrics = val_metrics
                
                # Save Best Model Checkpoint
                best_save_path = os.path.join(base_output_dir, f'model_fold_{fold+1}.pth') 
                torch.save(model.state_dict(), best_save_path)
                print(f"  [New Best] Fold {fold+1} Acc: {best_acc:.2f}% saved.")
                
                # Save Validation Predictions for Best Model
                valid_pred_path = os.path.join(base_output_dir, f'{fold+1}_valid_best.csv')
                with open(valid_pred_path, 'w') as f_pred:
                    f_pred.write("Label,Probability\n")
                    for t, p in zip(val_targets, val_probs):
                        f_pred.write(f"{int(t)},{p:.6f}\n")

        # --- End of Fold: Write Best Metrics to Log ---
        if best_metrics is not None:
            all_folds_metrics.append(best_metrics)
            with open(log_path, 'a') as f:
                f.write(f"{fold+1},{best_metrics['ACC']:.4f},{best_metrics['Sn']:.4f},{best_metrics['Sp']:.4f},"
                        f"{best_metrics['Pre']:.4f},{best_metrics['F1']:.4f},{best_metrics['MCC']:.4f},{best_metrics['AUC']:.4f}\n")

    # --- End of All Folds: Write Average ---
    print("\n" + "="*30)
    print("Cross Validation Complete")
    
    if len(all_folds_metrics) > 0:
        avg_metrics = {}
        for key in ['ACC', 'Sn', 'Sp', 'Pre', 'F1', 'MCC', 'AUC']:
            avg_metrics[key] = np.mean([m[key] for m in all_folds_metrics])
            
        with open(log_path, 'a') as f:
            f.write(f"Average,{avg_metrics['ACC']:.4f},{avg_metrics['Sn']:.4f},{avg_metrics['Sp']:.4f},"
                    f"{avg_metrics['Pre']:.4f},{avg_metrics['F1']:.4f},{avg_metrics['MCC']:.4f},{avg_metrics['AUC']:.4f}\n")
        
        print(f"Average Accuracy: {avg_metrics['ACC']:.2f}%")
        print(f"Average AUC:      {avg_metrics['AUC']:.4f}")

    return None


def validate_full(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            d_input = (batch['drug_input'][0].to(device), batch['drug_input'][1].to(device), batch['drug_input'][2].to(device))
            p_input = (batch['prot_input'][0].to(device), batch['prot_input'][1].to(device), batch['prot_input'][2].to(device))
            labels = batch['labels'].to(device).unsqueeze(1)
            
            outputs = model(d_input, p_input)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    # Compute Metrics
    all_targets = np.array(all_targets).flatten()
    all_probs = np.array(all_probs).flatten()
    all_preds = (all_probs > 0.5).astype(int)
    
    acc = accuracy_score(all_targets, all_preds) * 100
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.5
        
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel() if len(np.unique(all_targets)) > 1 else (0,0,0,0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall / Sn
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 # Sp
    precision = precision_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_targets, all_preds)
    
    metrics = {
        'Loss': val_loss / len(loader),
        'ACC': acc,
        'Sn': sensitivity,
        'Sp': specificity,
        'Pre': precision,
        'F1': f1,
        'MCC': mcc,
        'AUC': auc
    }
    
    return metrics, all_preds, all_targets, all_probs
