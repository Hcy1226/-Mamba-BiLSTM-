import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
    
    def forward(self, x, mask=None):
        # x: (B, L, D)
        # mask: (B, L)
        w = self.attention(x).squeeze(-1) # (B, L)
        
        if mask is not None:
            # mask is 1 for valid, 0 for pad. masked_fill expects mask to be True for "fill"
            # so we fill where mask == 0
            w = w.masked_fill(mask == 0, -1e9)
        
        weights = F.softmax(w, dim=1).unsqueeze(-1) # (B, L, 1)
        return (x * weights).sum(dim=1)

# --- Layer 1: 输入与表征层 ---

class DrugEncoder(nn.Module):
    def __init__(self, smiles_model_name='seyonec/ChemBERTa-zinc-base-v1', 
                 graph_in_channels=74, graph_hidden_channels=128, 
                 out_channels=256, fine_tune=False):
        """
        药物编码器: 结合SMILES序列特征与分子图结构特征
        """
        super(DrugEncoder, self).__init__()
        
        # 1. 序列支路 (SMILES)
        self.bert = AutoModel.from_pretrained(smiles_model_name)
        
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 2. 结构支路 (Graph)
        self.gat1 = GATConv(graph_in_channels, graph_hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(graph_hidden_channels * 4, graph_hidden_channels, heads=1, concat=False)
        self.graph_pool = global_mean_pool
        
        # 3. 融合层
        self.fusion_dim = self.bert_hidden_size + graph_hidden_channels
        self.project = nn.Linear(self.fusion_dim, out_channels)
        
    def forward(self, input_ids, attention_mask, graph_data):
        # 序列特征
        seq_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # 结构特征
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        graph_output = self.graph_pool(x, batch)
        
        # 融合: 将图特征扩展并拼接到序列每个token
        batch_size, seq_len, _ = seq_output.size()
        graph_expanded = graph_output.unsqueeze(1).expand(-1, seq_len, -1)
        fused_features = torch.cat([seq_output, graph_expanded], dim=-1)
        
        return self.project(fused_features)

class ProteinEncoder(nn.Module):
    def __init__(self, esm_model_name='facebook/esm2_t6_8M_UR50D',
                 graph_in_channels=1280, graph_hidden_channels=256,
                 out_channels=512, fine_tune=False):
        """
        蛋白质编码器: 结合氨基酸序列特征与残基接触图特征
        """
        super(ProteinEncoder, self).__init__()
        
        # 1. 序列支路 (ESM-2)
        self.esm = AutoModel.from_pretrained(esm_model_name)
        
        if not fine_tune:
            for param in self.esm.parameters():
                param.requires_grad = False
        
        self.esm_hidden_size = self.esm.config.hidden_size
        
        # 2. 结构支路 (Contact Map / Structure using GCN)
        self.gcn1 = GCNConv(self.esm_hidden_size, graph_hidden_channels)
        self.gcn2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        
        # 3. 融合层
        self.fusion_dim = self.esm_hidden_size + graph_hidden_channels
        self.project = nn.Linear(self.fusion_dim, out_channels)

    def forward(self, input_ids, attention_mask, edge_index):
        # 序列特征
        seq_output = self.esm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # 结构特征
        batch_size, seq_len, hidden_dim = seq_output.size()
        flat_x = seq_output.view(-1, hidden_dim) 
        
        gcn_x = F.relu(self.gcn1(flat_x, edge_index))
        gcn_x = F.relu(self.gcn2(gcn_x, edge_index))
        struct_output = gcn_x.view(batch_size, seq_len, -1)
        
        # 融合
        fused_features = torch.cat([seq_output, struct_output], dim=-1)
        
        return self.project(fused_features)

# --- Layer 2: 混合特征提取层 (Mamba-BiLSTM) ---

# --- Layer 2: 混合特征提取层 (Mamba-BiLSTM) ---

# 尝试导入 Mamba，如果失败则提供提示
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("Warning: mamba_ssm not found. Please install it for Mamba models.")

class MambaBlock(nn.Module):
    """
    Real Mamba Block with Residual Connection and RMSNorm/LayerNorm.
    Structure: Input -> Norm -> Mamba -> Residual -> Output
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        if not HAS_MAMBA:
             raise ImportError("mamba_ssm is required for this model. Run `pip install mamba-ssm`.")
        
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            use_fast_path=True # 使用 CUDA 优化路径
        )
    
    def forward(self, x):
        # x: (Batch, Seq, Dim)
        # Residual Connection: x + Mamba(Norm(x))
        return x + self.mamba(self.norm(x))

class MambaBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mamba_layers=3):
        super(MambaBiLSTM, self).__init__()
        
        # 1. Multi-Layer Mamba Stack (Residual included in blocks)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(dim=input_dim) for _ in range(num_mamba_layers)
        ])
        
        # 2. BiLSTM Layer (Local Feature Extraction)
        # Mamba 擅长捕捉长距离依赖，BiLSTM 增强局部上下文
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.norm_final = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Input_Dim)
        
        # Pass through Mamba Layers
        x_mamba = x
        for layer in self.mamba_layers:
            x_mamba = layer(x_mamba)
            
        # Pass through BiLSTM
        x_lstm, _ = self.bilstm(x_mamba)
        
        # Final Residual Fusion (Mamba + BiLSTM)
        # 此时 x_mamba 是多层 Mamba 的输出，包含了全局长程信息
        # x_lstm 是 BiLSTM 的输出，包含了双向局部信息
        return self.norm_final(x_lstm + x_mamba)

# --- Layer 3: 双向交互注意力层 (Bi-directional Attention) ---

class BidirectionalAttention(nn.Module):
    def __init__(self, dim):
        super(BidirectionalAttention, self).__init__()
        # Drug 查询 Protein (Q=Drug, K=Prot, V=Prot) -> 获取与Drug相关的Prot信息
        self.cross_attn_d2p = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        
        # Protein 查询 Drug (Q=Prot, K=Drug, V=Drug) -> 获取与Protein相关的Drug信息
        self.cross_attn_p2d = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, drug_feat, prot_feat, drug_mask=None, prot_mask=None):
        """
        drug_feat: (B, Ld, D)
        prot_feat: (B, Lp, D)
        Return:
            d_context: 增强后的药物特征
            p_context: 增强后的蛋白质特征
            attn_maps: (d2p_map, p2d_map)
        """
        # Drug -> Protein
        # key_padding_mask 需要是 boolean: True 表示被 mask (忽略)
        # 我们的 dataset 返回 mask是 1.0 (valid) / 0.0 (padding)，需要取反
        # masked_fill 需要 ByteTensor 或 BoolTensor
        
        # 注意 MultiheadAttention mask: 
        # key_padding_mask: (N, S) where True indicates elements to ignore
        
        d_pad_mask = (drug_mask == 0) if drug_mask is not None else None
        p_pad_mask = (prot_mask == 0) if prot_mask is not None else None

        # D queries P
        d_context, d2p_weights = self.cross_attn_d2p(
            query=drug_feat, 
            key=prot_feat, 
            value=prot_feat, 
            key_padding_mask=p_pad_mask
        )
        
        # P queries D
        p_context, p2d_weights = self.cross_attn_p2d(
            query=prot_feat, 
            key=drug_feat, 
            value=drug_feat, 
            key_padding_mask=d_pad_mask
        )
        
        return d_context, p_context, (d2p_weights, p2d_weights)

# --- Layer 4 + Total Model ---

class MambaBiLSTMModel(nn.Module):
    def __init__(self, drug_dim=256, prot_dim=512, hidden_dim=256, fine_tune=False):
        super(MambaBiLSTMModel, self).__init__()
        
        # Layer 1: Encoding
        self.drug_encoder = DrugEncoder(out_channels=drug_dim, fine_tune=fine_tune)
        self.prot_encoder = ProteinEncoder(out_channels=prot_dim, fine_tune=fine_tune)
        
        # Projection
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        
        # Layer 2: Mamba + BiLSTM
        self.drug_process = MambaBiLSTM(hidden_dim, hidden_dim, num_mamba_layers=3)
        self.prot_process = MambaBiLSTM(hidden_dim, hidden_dim, num_mamba_layers=3)
        
        # Layer 3: Bidirectional Attention
        self.bi_attention = BidirectionalAttention(hidden_dim)
        
        # Layer 4: Global Pooling (Attention Pooling)
        self.drug_pool = AttentionPooling(hidden_dim)
        self.prot_pool = AttentionPooling(hidden_dim)
        
        # Layer 5: Prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, drug_input, prot_input):
        d_emb = self.drug_encoder(*drug_input)
        p_emb = self.prot_encoder(*prot_input)
        
        d_feat = self.drug_proj(d_emb)
        p_feat = self.prot_proj(p_emb)
        
        d_feat = self.drug_process(d_feat)
        p_feat = self.prot_process(p_feat)
        
        d_mask = drug_input[1]
        p_mask = prot_input[1]
        
        d_context, p_context, attn_maps = self.bi_attention(d_feat, p_feat, d_mask, p_mask)
        
        # Weighted Attention Pooling
        d_global = self.drug_pool(d_context, d_mask)
        p_global = self.prot_pool(p_context, p_mask)
        
        combined = torch.cat([d_global, p_global], dim=-1)
        score = self.classifier(combined)
        
        return score, attn_maps

class DeepDTAModel(nn.Module):
    """
    DeepDTA Variant: Uses CNNs on top of embedding sequences.
    """
    def __init__(self, drug_dim=256, prot_dim=512, hidden_dim=256):
        super(DeepDTAModel, self).__init__()
        self.drug_encoder = DrugEncoder(out_channels=drug_dim)
        self.prot_encoder = ProteinEncoder(out_channels=prot_dim)
        
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        
        # CNN Layers (1D Conv)
        self.drug_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.prot_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, 128), # hidden*2 + hidden*2
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_input, prot_input):
        d_emb = self.drug_proj(self.drug_encoder(*drug_input)) # (B, L, H)
        p_emb = self.prot_proj(self.prot_encoder(*prot_input)) # (B, L, H)
        
        # Permute for CNN (B, H, L)
        d_conv = self.drug_cnn(d_emb.permute(0, 2, 1)).squeeze(-1) # (B, 2H)
        p_conv = self.prot_cnn(p_emb.permute(0, 2, 1)).squeeze(-1) # (B, 2H)
        
        combined = torch.cat([d_conv, p_conv], dim=-1)
        score = self.classifier(combined)
        return score, None

class TransformerDTIModel(nn.Module):
    """
    Transformer Variant: Uses standard Transformer Encoders.
    """
    def __init__(self, drug_dim=256, prot_dim=512, hidden_dim=256):
        super(TransformerDTIModel, self).__init__()
        self.drug_encoder = DrugEncoder(out_channels=drug_dim)
        self.prot_encoder = ProteinEncoder(out_channels=prot_dim)
        
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.drug_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.prot_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_input, prot_input):
        d_emb = self.drug_proj(self.drug_encoder(*drug_input))
        p_emb = self.prot_proj(self.prot_encoder(*prot_input))
        
        d_trans = self.drug_transformer(d_emb)
        p_trans = self.prot_transformer(p_emb)
        
        # Global Average Pooling
        d_global = d_trans.mean(dim=1)
        p_global = p_trans.mean(dim=1)
        
        combined = torch.cat([d_global, p_global], dim=-1)
        score = self.classifier(combined)
        return score, None

class GraphDTAModel(nn.Module):
    """
    GraphDTA Variant: Focuses on Graph features (simulated by pooling immediately).
    Bypasses sequence modeling.
    """
    def __init__(self, drug_dim=256, prot_dim=512, hidden_dim=256):
        super(GraphDTAModel, self).__init__()
        self.drug_encoder = DrugEncoder(out_channels=drug_dim)
        self.prot_encoder = ProteinEncoder(out_channels=prot_dim)
        
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_input, prot_input):
        d_emb = self.drug_proj(self.drug_encoder(*drug_input))
        p_emb = self.prot_proj(self.prot_encoder(*prot_input))
        
        # Max Pooling (Readout)
        d_global = d_emb.max(dim=1)[0]
        p_global = p_emb.max(dim=1)[0]
        
        combined = torch.cat([d_global, p_global], dim=-1)
        score = self.classifier(combined)
        return score, None

class MCANetModel(nn.Module):
    """
    MCANet Variant: CNN Encoders + Cross-Attention.
    """
    def __init__(self, drug_dim=256, prot_dim=512, hidden_dim=256):
        super(MCANetModel, self).__init__()
        self.drug_encoder = DrugEncoder(out_channels=drug_dim)
        self.prot_encoder = ProteinEncoder(out_channels=prot_dim)
        
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.prot_proj = nn.Linear(prot_dim, hidden_dim)
        
        # CNN Blocks (Feature Extraction)
        self.drug_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.prot_cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Multihead Cross-Attention
        self.bi_attention = BidirectionalAttention(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, drug_input, prot_input):
        d_emb = self.drug_proj(self.drug_encoder(*drug_input))
        p_emb = self.prot_proj(self.prot_encoder(*prot_input))
        
        # CNN Extraction
        d_feat = self.drug_cnn(d_emb.permute(0, 2, 1)).permute(0, 2, 1) # (B, L, H)
        p_feat = self.prot_cnn(p_emb.permute(0, 2, 1)).permute(0, 2, 1) # (B, L, H)
        
        d_mask = drug_input[1]
        p_mask = prot_input[1]
        
        # Cross Attention
        d_context, p_context, _ = self.bi_attention(d_feat, p_feat, d_mask, p_mask)
        
        # Global Pooling
        d_global = (d_context * d_mask.unsqueeze(-1)).sum(dim=1) / d_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        p_global = (p_context * p_mask.unsqueeze(-1)).sum(dim=1) / p_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        
        combined = torch.cat([d_global, p_global], dim=-1)
        score = self.classifier(combined)
        return score, None

def get_model(model_name, **kwargs):
    name = model_name.lower()
    if name == 'mamba_bilstm':
        return MambaBiLSTMModel(**kwargs)
    elif name == 'deepdta':
        return DeepDTAModel(**kwargs)
    elif name == 'transformer':
        return TransformerDTIModel(**kwargs)
    elif name == 'graphdta':
        return GraphDTAModel(**kwargs)
    elif name == 'mcanet':
        return MCANetModel(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

