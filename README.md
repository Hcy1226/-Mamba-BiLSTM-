# 基于 Mamba 与 BiLSTM 混合架构的药物-靶点相互作用预测 (Mamba-BiLATT DTI)

本项目实现了一种基于深度学习的药物-靶点相互作用（DTI）预测模型，结合了 **Mamba (State Space Model)** 的全局建模能力与 **BiLSTM** 的局部特征提取能力，并引入了双向交互注意力机制（Bi-directional Attention）来捕捉药物与蛋白质之间的关键结合特征。

## 📂 核心文件结构

本项目经过精简，核心逻辑由以下 5 个 Python 文件组成：

| 文件名 | 描述 | 对应架构层级 |
| :--- | :--- | :--- |
| **`dataset.py`** | **数据处理**。负责加载 Davis 等数据集，对 SMILES 进行 Tokenization 和图构建，对蛋白质序列进行编码。 | 第一层：输入与表征层 |
| **`model.py`** | **模型定义**。包含 DrugEncoder, ProteinEncoder, Mamba-BiLSTM 模块, Bi-Attention 以及预测头。 | 第二、三、四层 |
| **`train.py`** | **训练脚本**。定义了训练循环、验证过程、损失函数计算以及模型保存逻辑。 | - |
| **`test.py`** | **测试脚本**。用于加载训练好的权重文件，在独立测试集上评估模型性能 (Loss, Accuracy)。 | - |
| **`run.py`** | **主运行入口**。封装了命令行接口 (CLI)，用于一键启动训练或测试任务。 | - |

此外：
*   `arch.md`: 详细的模型架构设计文档。
*   `data/`: 存放数据集（如 `Davis.txt`）。

## 🛠️ 环境安装 (Installation)

由于本项目依赖 `rdkit` 和 `transformers`，建议在 **Anaconda** 环境中运行以避免编译错误。

### 推荐步骤

1.  **创建新环境** (Python 3.8):
    ```bash
    conda create -n dti_env python=3.8
    conda activate dti_env
    ```

2.  **安装 PyTorch**:
    *   请根据您的 CUDA 版本访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取安装命令。例如：
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3.  **安装核心依赖** (Transformers, Geometric, RDKit):
    ```bash
    pip install transformers torch-geometric matplotlib
    pip install rdkit-pypi  # 使用预编译的 RDKit
    ```

> **注意**: 如果遇到 `ImportError: packaging` 或 `ImportError: transformers` 相关错误，请尝试使用 conda 安装 transformer:
> `conda install -c huggingface transformers`

## 🚀 快速运行 (Usage)

所有操作均可通过 `run.py` 脚本执行。

### 1. 训练模型 (Training)
默认使用 `data/Davis.txt` 数据集。

```bash
# 正常训练 (默认参数: Batch Size=8, Epochs=10)
python run.py train

# 指定参数训练
python run.py train --epochs 50 --batch_size 16 --lr 0.0001
```

**调试模式 (Debug)**:
如果想快速测试代码是否跑通（只用极少量数据）：
```bash
python run.py train --debug
```

### 2. 测试模型 (Testing)
加载训练好的权重文件进行评估。

```bash
# 假设权重保存在 checkpoints 文件夹中
python run.py test --weights checkpoints/model_epoch_10.pth
```
