import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from mamba_ssm import Mamba

# --- Layer 1: 输入与表征层 ---

class DrugEncoder(nn.Module):
    def __init__(self, smiles_model_name='seyonec/ChemBERTa-zinc-base-v1', 
                 graph_in_channels=74, graph_hidden_channels=128, 
                 out_channels=256, fine_tune=False):
        super(DrugEncoder, self).__init__()
        
        # 1. 序列支路 (SMILES)
        self.bert = AutoModel.from_pretrained(smiles_model_name)
        
        if not fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.bert_hidden_size = self.bert.config.hidden_size
        
        # 2. 结构支路 (Graph)
        self.gat1 = GATConv(graph_in_channels, graph_hidden_channels)
        self.gat2 = GATConv(graph_hidden_channels, graph_hidden_channels)
        self.graph_proj = nn.Linear(graph_hidden_channels, out_channels)
        
        # 融合
        self.fusion = nn.Linear(self.bert_hidden_size + out_channels, out_channels)
        
    def forward(self, input_ids, attention_mask, graph_data):
        # 序列特征
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq_feat = bert_output.last_hidden_state # (B, L, D)
        
        # 图特征
        x, edge_index, batch = graph_data.x, graph_data.edge_index, graph_data.batch
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        graph_feat = global_mean_pool(x, batch) # (B, D_g)
        graph_feat = self.graph_proj(graph_feat) # (B, Out)
        
        # 扩展图特征以匹配序列长度 (简单的拼接，后续可以通过Attention改进)
        graph_feat_expanded = graph_feat.unsqueeze(1).repeat(1, seq_feat.size(1), 1)
        
        # 拼接
        combined = torch.cat([seq_feat, graph_feat_expanded], dim=-1)
        return self.fusion(combined)

class ProteinEncoder(nn.Module):
    def __init__(self, esm_model_name='facebook/esm2_t6_8M_UR50D',
                 graph_in_channels=1280, graph_hidden_channels=256,
                 out_channels=512, fine_tune=False):
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
        self.graph_proj = nn.Linear(graph_hidden_channels, out_channels)
        
        self.fusion = nn.Linear(self.esm_hidden_size + out_channels, out_channels)
        
    def forward(self, input_ids, attention_mask, edge_index):
        # 序列特征
        esm_output = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        seq_feat = esm_output.last_hidden_state
        
        # 结构特征 (Node features are seq features)
        # Flatten batch for GCN
        # (B, L, D) -> (B*L, D)
        B, L, D = seq_feat.size()
        x_flat = seq_feat.view(-1, D)
        
        # Note: edge_index is already batched in collate_fn
        x_gcn = F.relu(self.gcn1(x_flat, edge_index))
        x_gcn = F.relu(self.gcn2(x_gcn, edge_index))
        
        x_gcn = x_gcn.view(B, L, -1)
        graph_feat = self.graph_proj(x_gcn)
        
        combined = torch.cat([seq_feat, graph_feat], dim=-1)
        return self.fusion(combined)

# --- Layer 2: 上下文建模层 (Mamba + BiLSTM) ---

class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=dim, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.mamba(x))

class MambaBiLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_mamba_layers=3):
        super(MambaBiLSTM, self).__init__()
        
        self.mamba_layers = nn.ModuleList([
            MambaBlock(in_dim) for _ in range(num_mamba_layers)
        ])
        
        self.bilstm = nn.LSTM(in_dim, hidden_dim // 2, num_layers=1, 
                              bidirectional=True, batch_first=True)
        
        self.fusion = nn.Linear(in_dim + hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Mamba Path
        m_out = x
        for layer in self.mamba_layers:
            m_out = layer(m_out)
            
        # BiLSTM Path
        l_out, _ = self.bilstm(x)
        
        # Residual Fusion
        combined = torch.cat([m_out, l_out], dim=-1)
        return self.fusion(combined)

# --- Layer 3: 交互层 ---

class BidirectionalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BidirectionalAttention, self).__init__()
        self.W_d = nn.Linear(hidden_dim, hidden_dim)
        self.W_p = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
    def forward(self, drug_feat, prot_feat, drug_mask, prot_mask):
        # drug_feat: (B, Ld, H)
        # prot_feat: (B, Lp, H)
        
        Q = self.W_d(drug_feat)
        K = self.W_p(prot_feat)
        
        # Attention Scores (B, Ld, Lp)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale.to(drug_feat.device)
        
        # Masking
        # mask shape: (B, L) -> (B, L, 1) or (B, 1, L)
        d_mask = drug_mask.unsqueeze(2).float() # (B, Ld, 1)
        p_mask = prot_mask.unsqueeze(1).float() # (B, 1, Lp)
        
        mask = torch.bmm(d_mask, p_mask) # (B, Ld, Lp)
        scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_d2p = F.softmax(scores, dim=2) # drug attends to prot
        attn_p2d = F.softmax(scores, dim=1) # prot attends to drug
        
        # Context
        # context_d: drug 结合了 protein 信息
        context_d = torch.bmm(attn_d2p, prot_feat) # (B, Ld, H)
        # context_p: protein 结合了 drug 信息 (transpose weights for dim 1 sum to 1)
        context_p = torch.bmm(attn_p2d.transpose(1, 2), drug_feat) # (B, Lp, H)
        
        return context_d, context_p, scores

# --- Layer 4 + Total Model ---

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
    
    def forward(self, x, mask=None):
        w = self.attention(x).squeeze(-1)
        if mask is not None:
            w = w.masked_fill(mask == 0, -1e9)
        weights = F.softmax(w, dim=1).unsqueeze(-1)
        return (x * weights).sum(dim=1)

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
        # Stacked Mambas for deeper reasoning
        self.drug_process = MambaBiLSTM(hidden_dim, hidden_dim, num_mamba_layers=3)
        self.prot_process = MambaBiLSTM(hidden_dim, hidden_dim, num_mamba_layers=3)
        
        # Layer 3: Bidirectional Attention
        self.bi_attention = BidirectionalAttention(hidden_dim)
        
        # Global Pooling
        self.d_pool = AttentionPooling(hidden_dim)
        self.p_pool = AttentionPooling(hidden_dim)
        
        # Layer 4: Prediction (Robust Interaction Head)
        # Logits output (No Sigmoid) for BCEWithLogitsLoss
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256), # Concatenation + Interaction
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, drug_input, prot_input):
        d_ids, d_mask, d_graph = drug_input
        p_ids, p_mask, p_edge = prot_input
        
        # 1. Encode
        d_feat = self.drug_encoder(d_ids, d_mask, d_graph)
        p_feat = self.prot_encoder(p_ids, p_mask, p_edge)
        
        # Project
        d_feat = F.relu(self.drug_proj(d_feat))
        p_feat = F.relu(self.prot_proj(p_feat))
        
        # 2. Sequential Process
        d_feat = self.drug_process(d_feat)
        p_feat = self.prot_process(p_feat)
        
        # 3. Bi-Interaction
        d_context, p_context, _ = self.bi_attention(d_feat, p_feat, d_mask, p_mask)
        
        # 4. Global Pooling
        # Fusion of original feature + context
        d_final = d_feat + d_context
        p_final = p_feat + p_context
        
        d_vec = self.d_pool(d_final, d_mask)
        p_vec = self.p_pool(p_final, p_mask)
        
        # 5. Classifier
        # Interaction feature: Element-wise product
        interaction = d_vec * p_vec
        
        # Concat: [Drug, Protein, Drug*Protein]
        combined = torch.cat([d_vec, p_vec, interaction], dim=-1)
        
        return self.classifier(combined)

# --- Baseline Models ---

class DeepDTA(nn.Module):
    """
    DeepDTA: Deep Drug-Target Binding Affinity Prediction
    Uses CNNs for both Drug (SMILES) and Protein (Sequence)
    """
    def __init__(self, drug_vocab_size=600, prot_vocab_size=26, 
                 embedding_dim=128, hidden_dim=256):
        super(DeepDTA, self).__init__()
        
        # Drug Branch (CNN)
        self.drug_embed = nn.Embedding(drug_vocab_size, embedding_dim)
        self.drug_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=6),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=8),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # Protein Branch (CNN)
        self.prot_embed = nn.Embedding(prot_vocab_size, embedding_dim)
        self.prot_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(64, 96, kernel_size=12),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # FC Layers
        self.fc = nn.Sequential(
            nn.Linear(96 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, drug_input, prot_input):
        # drug_input -> (ids, mask, graph) -> we only use ids
        d_ids = drug_input[0] 
        # prot_input -> (ids, mask, edge) -> we only use ids
        p_ids = prot_input[0]
        
        # Embed: (B, L) -> (B, L, E) -> (B, E, L) for Conv1d
        d_x = self.drug_embed(d_ids).permute(0, 2, 1)
        p_x = self.prot_embed(p_ids).permute(0, 2, 1)
        
        d_feat = self.drug_cnn(d_x).squeeze(-1) # (B, 96)
        p_feat = self.prot_cnn(p_x).squeeze(-1) # (B, 96)
        
        combined = torch.cat([d_feat, p_feat], dim=1)
        return self.fc(combined)


class TransformerCPI(nn.Module):
    """
    TransformerCPI: Transformer-based Chem-Prot Interaction
    Uses simplified Transformer Encoders for both inputs
    """
    def __init__(self, drug_vocab_size=600, prot_vocab_size=26, 
                 d_model=128, nhead=4, num_layers=2):
        super(TransformerCPI, self).__init__()
        
        self.drug_embed = nn.Embedding(drug_vocab_size, d_model)
        self.prot_embed = nn.Embedding(prot_vocab_size, d_model)
        
        self.drug_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, batch_first=True),
            num_layers
        )
        
        self.prot_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, batch_first=True),
            num_layers
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, drug_input, prot_input):
        d_ids, d_mask = drug_input[0], drug_input[1]
        p_ids, p_mask = prot_input[0], prot_input[1]
        
        d_x = self.drug_embed(d_ids) # (B, L, D)
        p_x = self.prot_embed(p_ids)
        
        # Pass masks (invert logic for Transformer usually, but PyTorch handle src_key_padding_mask as True=ignore)
        # Our mask is 1=valid, 0=pad. So we need ~mask.bool()
        d_padding_mask = (d_mask == 0)
        p_padding_mask = (p_mask == 0)
        
        d_out = self.drug_encoder(d_x, src_key_padding_mask=d_padding_mask)
        p_out = self.prot_encoder(p_x, src_key_padding_mask=p_padding_mask)
        
        # Global Mean Pooling (ignoring pads) -> simplified to simple mean for baseline
        d_vec = d_out.mean(dim=1)
        p_vec = p_out.mean(dim=1)
        
        combined = torch.cat([d_vec, p_vec], dim=1)
        return self.classifier(combined)


class MCANet(nn.Module):
    """
    MCANet: Multi-source Co-Attention Network (Simplified/Representative)
    Uses Co-Attention between Drug and Protein features
    """
    def __init__(self, drug_vocab_size=600, prot_vocab_size=26, hidden_dim=128):
        super(MCANet, self).__init__()
        
        self.drug_embed = nn.Embedding(drug_vocab_size, hidden_dim)
        self.prot_embed = nn.Embedding(prot_vocab_size, hidden_dim)
        
        # Co-Attention Weight Matrices
        self.W_d = nn.Linear(hidden_dim, hidden_dim)
        self.W_p = nn.Linear(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, drug_input, prot_input):
        d_ids = drug_input[0]
        p_ids = prot_input[0]
        
        d_feat = self.drug_embed(d_ids) # (B, Ld, H)
        p_feat = self.prot_embed(p_ids) # (B, Lp, H)
        
        # Valid Co-Attention (Baseline impl)
        # Affinity Matrix
        aff = torch.bmm(d_feat, p_feat.transpose(1, 2)) # (B, Ld, Lp)
        
        # Attention maps
        att_d = F.softmax(aff.max(dim=2)[0], dim=1).unsqueeze(2) # (B, Ld, 1)
        att_p = F.softmax(aff.max(dim=1)[0], dim=1).unsqueeze(2) # (B, Lp, 1)
        
        # Weighted Sum
        d_vec = (d_feat * att_d).sum(dim=1) # (B, H)
        p_vec = (p_feat * att_p).sum(dim=1) # (B, H)
        
        combined = torch.cat([d_vec, p_vec], dim=1)
        return self.classifier(combined)


# --- Factory Function ---
def get_model(model_name, **kwargs):
    model_name = model_name.lower().replace('-', '').replace('_', '')
    
    if model_name == 'mambabilstm':
        return MambaBiLSTMModel(fine_tune=kwargs.get('fine_tune', False), 
                                hidden_dim=kwargs.get('hidden_dim', 256))
    elif 'deep' in model_name: # DeepDTA, DeepConv-DTI
        return DeepDTA(drug_vocab_size=kwargs.get('drug_vocab_size', 600),
                       prot_vocab_size=kwargs.get('prot_vocab_size', 26))
    elif 'transformer' in model_name: # TransformerCPI
        return TransformerCPI(drug_vocab_size=kwargs.get('drug_vocab_size', 600),
                              prot_vocab_size=kwargs.get('prot_vocab_size', 26))
    elif 'mcanet' in model_name: # MCANet
        return MCANet(drug_vocab_size=kwargs.get('drug_vocab_size', 600),
                      prot_vocab_size=kwargs.get('prot_vocab_size', 26))
    else:
        raise ValueError(f"Unknown model name: {model_name}")
