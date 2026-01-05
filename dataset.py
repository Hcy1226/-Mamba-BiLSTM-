import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
import numpy as np
from rdkit import Chem
from tqdm import tqdm

# 这是连接原始数据文件与深度学习模型的桥梁，负责将文本数据转换为模型能看懂的张量（Tensors）。
class DTIDataset(Dataset):
    def __init__(self, file_path, smi_tokenizer, prot_tokenizer, max_len_drug=128, max_len_prot=512):
        """
        DTI 数据集加载器 (优化版: 预处理所有数据到内存)
        """
        self.smi_tokenizer = smi_tokenizer
        self.prot_tokenizer = prot_tokenizer
        self.max_len_drug = max_len_drug
        self.max_len_prot = max_len_prot
        
        # 1. 读取原始数据
        raw_data = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            label = float(parts[-1])
                            seq = parts[-2]
                            smiles = parts[-3]
                            raw_data.append({'smiles': smiles, 'sequence': seq, 'label': label})
                        except ValueError:
                            continue
        except FileNotFoundError:
            print(f"Warning: File {file_path} not found.")

        print(f"Pre-processing {len(raw_data)} samples...")
        
        # 2. 批量 Tokenization (极速提升)
        # 提取列表
        smiles_list = [d['smiles'] for d in raw_data]
        seq_list = [d['sequence'] for d in raw_data]
        self.labels = [d['label'] for d in raw_data]
        
        # Drug Tokenization
        print("Tokenizing Drugs...")
        drug_enc = self.smi_tokenizer(smiles_list, padding='max_length', truncation=True, max_length=max_len_drug, return_tensors='pt')
        self.drug_ids = drug_enc['input_ids']
        self.drug_masks = drug_enc['attention_mask']
        
        # Protein Tokenization
        print("Tokenizing Proteins...")
        prot_enc = self.prot_tokenizer(seq_list, padding='max_length', truncation=True, max_length=max_len_prot, return_tensors='pt')
        self.prot_ids = prot_enc['input_ids']
        self.prot_masks = prot_enc['attention_mask']
        
        # 3. 预计算分子图 (避免训练时重复计算)
        print("Generating Molecular Graphs...")
        self.drug_graphs = []
        for smi in tqdm(smiles_list, desc="Graphs"):
            g = self._get_molecule_graph(smi)
            if g is None:
                g = Data(x=torch.zeros((1, 74)), edge_index=torch.empty((2, 0), dtype=torch.long))
            self.drug_graphs.append(g)
            
    def __len__(self):
        return len(self.labels)

    def _get_molecule_graph(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        
        atom_features = []
        for atom in mol.GetAtoms():
            features = np.zeros(74, dtype=np.float32)
            features[0] = atom.GetAtomicNum()
            features[1] = atom.GetDegree()
            features[2] = atom.GetFormalCharge()
            atom_features.append(features)
        x = torch.tensor(np.array(atom_features), dtype=torch.float32)
        
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])
        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def __getitem__(self, idx):
        # 极速读取: 直接从内存取张量
        real_prot_len = self.prot_masks[idx].sum().item()
        
        return {
            'drug_ids': self.drug_ids[idx],
            'drug_mask': self.drug_masks[idx],
            'drug_graph': self.drug_graphs[idx],
            'prot_ids': self.prot_ids[idx],
            'prot_mask': self.prot_masks[idx],
            'prot_len': real_prot_len,
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

    def slice_subset(self, n):
        """
        Debug模式专用：只保留前 n 条数据
        """
        n = min(n, len(self.labels))
        self.labels = self.labels[:n]
        self.drug_ids = self.drug_ids[:n]
        self.drug_masks = self.drug_masks[:n]
        self.drug_graphs = self.drug_graphs[:n]
        self.prot_ids = self.prot_ids[:n]
        self.prot_masks = self.prot_masks[:n]

def collate_dti(batch):
    """
    自定义 Collate 函数
    处理 PyG Batch 对象 和 动态构建 Protein Edge Index
    """
    drug_ids = torch.stack([item['drug_ids'] for item in batch])
    drug_mask = torch.stack([item['drug_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    # Batch Drug Graph
    drug_graphs = [item['drug_graph'] for item in batch]
    batched_drug_graph = Batch.from_data_list(drug_graphs)
    
    prot_ids = torch.stack([item['prot_ids'] for item in batch])
    prot_mask = torch.stack([item['prot_mask'] for item in batch])
    
    # 构建 Batched Protein Edge Index (简单的线性连接)
    batch_size = len(batch)
    max_len = prot_ids.size(1)
    
    all_edges = []
    for i, item in enumerate(batch):
        offset = i * max_len
        l = item['prot_len']
        
        if l > 1:
            source = torch.arange(0, l - 1, dtype=torch.long) + offset
            target = torch.arange(1, l, dtype=torch.long)     + offset
            
            edges = torch.stack([source, target], dim=0)
            all_edges.append(edges)
            edges_rev = torch.stack([target, source], dim=0)
            all_edges.append(edges_rev)
            
    if len(all_edges) > 0:
        prot_edge_index = torch.cat(all_edges, dim=1)
    else:
        prot_edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return {
        'drug_input': (drug_ids, drug_mask, batched_drug_graph),
        'prot_input': (prot_ids, prot_mask, prot_edge_index),
        'labels': labels
    }
