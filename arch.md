为了方便你写论文和画图，我为你梳理了一个符合 **SCI 高级期刊标准** 的总体架构图。这个架构兼顾了“序列+结构”的双模态输入，并完美嵌入了你的 **Mamba-BiLATT** 核心模块。

我们可以将架构分为四个层次：**输入与表征层**、**混合特征提取层**、**双向交互注意力层**、**输出预测层**。

---

### 总体架构图逻辑 (Mamba-BiLATT Overview)

#### 第一层：输入与表征层 (Input & Representation Layer)
这一层负责将原始的生物化学数据转换为高维向量。

*   **药物支路 (Drug Branch):**
    *   **序列输入:** SMILES $\rightarrow$ **ChemBERTa/Embedding** $\rightarrow$ 序列特征向量。
    *   **结构输入:** 分子图 (Graph) $\rightarrow$ **GAT/GIN (图神经网络)** $\rightarrow$ 结构特征向量。
    *   **融合:** 拼接（Concat）两部分特征，形成药物初始表征 $E_{drug}$。
*   **蛋白质支路 (Protein Branch):**
    *   **序列输入:** 氨基酸序列 $\rightarrow$ **ESM-2 (预训练编码器)** $\rightarrow$ 序列特征矩阵。
    *   **结构输入:** 接触图 (Contact Map/PDB) $\rightarrow$ **GCN (图卷积)** $\rightarrow$ 结构特征矩阵。
    *   **融合:** 拼接（Concat）特征，形成蛋白质初始表征 $E_{protein}$。

#### 第二层：混合特征提取层 (Hybrid Feature Extraction - Mamba-BiLSTM)
这是你的**核心贡献点**，药物和蛋白质特征分别通过该模块进行深度建模。

*   **Mamba Block (全局建模):**
    *   输入融合后的特征。利用其 **Selective Scan (S6)** 机制，在长序列（尤其是蛋白质）中捕捉全局长程依赖。
    *   *作用：* 建立序列远端残基之间的空间关联。
*   **BiLSTM Layer (局部细化):**
    *   接在 Mamba 之后。利用其双向循环机制，对邻近残基/原子的上下文进行精细化微调。
    *   *作用：* 弥补 SSM（状态空间模型）在局部特征平滑性上的不足。

#### 第三层：双向交互注意力层 (Bi-directional Attention Layer - BiLATT)
模拟药物与蛋白质“相互吸引、相互匹配”的过程。

*   **Cross-Attention (交叉注意力):**
    *   **Drug-to-Protein:** 药物分子去查询（Query）蛋白质序列，找到匹配的结合位点。
    *   **Protein-to-Drug:** 蛋白质残基反过来查询药物原子，识别关键官能团。
*   **Attention Map:** 生成一个热力图（Heatmap），这是论文中“可解释性”分析的关键。

#### 第四层：输出预测层 (Prediction Head)
*   **Feature Fusion:** 将经过注意力加权后的药物向量和蛋白质向量进行拼接。
*   **MLP (多层感知机):** 包含 Dropout 层防止过拟合。
*   **Output:** 
    *   分类任务：Sigmoid $\rightarrow$ 相互作用概率（0-1）。
    *   回归任务：Linear $\rightarrow$ 亲和力数值 ($pK_i, pK_d, pIC_{50}$)。

---
