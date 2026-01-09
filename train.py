import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


# 导入模块
from architectures import get_model
from dataset import DTIDataset, collate_dti

def train_model(data_path, output_dir='checkpoints', batch_size=8, epochs=10, lr=1e-4, folds=5, model_name='mamba_bilstm', fine_tune=False, hidden_dim=256, debug=False):
    """
    训练主函数 (5-Fold Cross Validation)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 1. 数据准备 ---
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
        batch_size = 8 # Slightly larger batch for 1000 samples
        epochs = 2 # Keep epochs small for quick test
        folds = 5  # Use full 5 folds to test the loop logic

    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # K-Fold Split
    kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
    
    # Metrics Storage
    all_folds_history = []
    
    print(f"Starting {folds}-Fold Cross Validation for Model: {model_name}...")
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{folds} ---")
        
        # Subsets
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler, collate_fn=collate_dti)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler, collate_fn=collate_dti)
        
        # --- 2. 模型初始化 (每个Fold重置模型) ---
        print(f"Initializing {model_name} (Fine-tune: {fine_tune}, Hidden: {hidden_dim})...")
        model = get_model(model_name, drug_dim=256, prot_dim=512, hidden_dim=hidden_dim, fine_tune=fine_tune).to(device)
        
        # --- 3. 训练配置 ---
        # 计算 pos_weight (负样本数 / 正样本数) 以处理类别不平衡
        all_labels = dataset.labels
        num_pos = sum(all_labels)
        num_neg = len(all_labels) - num_pos
        pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val]).to(device)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val:.2f} (Neg: {num_neg}, Pos: {num_pos})")
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'best_acc': 0.0}
        
        # --- 4. 训练循环 ---
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{epochs}", leave=False)
            for batch in loop:
                # Move to Device
                d_input = (batch['drug_input'][0].to(device), batch['drug_input'][1].to(device), batch['drug_input'][2].to(device))
                p_input = (batch['prot_input'][0].to(device), batch['prot_input'][1].to(device), batch['prot_input'][2].to(device))
                labels = batch['labels'].to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs, _ = model(d_input, p_input)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
                
            avg_loss = total_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            
            # --- 验证 ---
            val_result = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_result['loss'])
            history['val_acc'].append(val_result['acc'])
            
            print(f"  Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Loss={val_result['loss']:.4f}, Val Acc={val_result['acc']:.2f}%")
            
            # --- 写入 CSV 日志 ---
            log_path = os.path.join(output_dir, 'training_log.csv')
            with open(log_path, 'a') as f:
                if f.tell() == 0:
                    f.write("Fold,Epoch,Train_Loss,Val_Loss,Val_Acc\n")
                f.write(f"{fold+1},{epoch+1},{avg_loss:.5f},{val_result['loss']:.5f},{val_result['acc']:.4f}\n")
            
            # 记录最佳模型
            if val_result['acc'] > history['best_acc']:
                history['best_acc'] = val_result['acc']
                best_save_path = os.path.join(output_dir, f'model_fold_{fold+1}_best.pth')
                torch.save(model.state_dict(), best_save_path)
                print(f"  [New Best] Fold {fold+1} Best Model (Acc: {val_result['acc']:.2f}%) saved to {best_save_path}")

        # 保存该 Fold 的最终模型
        save_path = os.path.join(output_dir, f'model_fold_{fold+1}_last.pth')
        torch.save(model.state_dict(), save_path)
        
        all_folds_history.append(history)
        
    print("\n" + "="*30)
    print("Cross Validation Complete")
    
    # Calculate Average Performance
    avg_acc = np.mean([h['val_acc'][-1] for h in all_folds_history])
    print(f"Average Validation Accuracy: {avg_acc:.2f}%")
    
    return all_folds_history


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            d_input = (batch['drug_input'][0].to(device), batch['drug_input'][1].to(device), batch['drug_input'][2].to(device))
            p_input = (batch['prot_input'][0].to(device), batch['prot_input'][1].to(device), batch['prot_input'][2].to(device))
            labels = batch['labels'].to(device).unsqueeze(1)
            
            outputs, _ = model(d_input, p_input)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # 由于使用了 BCEWithLogitsLoss，输出是 logits，需要先 sigmoid 再判断
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
    return {
        'loss': val_loss / len(loader),
        'acc': 100 * correct / total if total > 0 else 0
    }
