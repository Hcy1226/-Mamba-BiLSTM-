import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
import argparse

from architectures import get_model
from dataset import DTIDataset, collate_dti
from train import validate_full

def test_model(weights_path, data_path, model_name='mamba_bilstm', batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing using device: {device}")
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights file {weights_path} not found.")
        return

    # Load Tokenizers
    print("Loading Tokenizers...")
    smi_tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    prot_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    
    # Load Data
    print("Loading Test Data...")
    # NOTE: In a real scenario, use a separate test file or split logic. 
    # Here we reuse the dataset class.
    dataset = DTIDataset(data_path, smi_tokenizer, prot_tokenizer, max_len_drug=128, max_len_prot=350)
    
    if len(dataset) == 0: 
        print("Test dataset is empty.")
        return

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_dti)
    
    # Load Model
    print(f"Loading Model ({model_name})...")
    model = get_model(model_name, drug_dim=256, prot_dim=512, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # Validate
    criterion = torch.nn.BCEWithLogitsLoss()
    metrics, preds, targets, probs = validate_full(model, test_loader, criterion, device)
    
    print("="*30)
    print("Test Results:")
    print(f"Loss: {metrics['Loss']:.4f}")
    print(f"Accuracy: {metrics['ACC']:.2f}%")
    print(f"AUC: {metrics['AUC']:.4f}")
    print(f"Sn: {metrics['Sn']:.4f}")
    print(f"Sp: {metrics['Sp']:.4f}")
    print(f"MCC: {metrics['MCC']:.4f}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to test data')
    args = parser.parse_args()
    
    test_model(args.weights, args.data)
