# PERT: Payload Encoding Representation from Transformer

这是论文 **"PERT: Payload Encoding Representation from Transformer for Encrypted Traffic Classification"** 的完整PyTorch实现。

## 论文简介

PERT是一种用于加密流量分类的新方法，它将基于Transformer的语言建模技术应用于网络数据包有效载荷。该方法分为两个阶段：

1. **预训练阶段**：在未标记的数据包上训练掩码语言模型（MLM），学习上下文表示
2. **微调阶段**：将预训练好的编码器用于流级别的分类任务

### 关键特点

- **Bigram标记化**：将字节对作为基本单位（词汇表大小：65536 + 3个特殊标记）
- **Transformer编码器**：使用多头注意力机制捕获上下文依赖关系
- **流级别分类**：每个流取前3个数据包，提取它们<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token后拼接分类
- **预训练+微调**：利用未标记数据提升性能

## 环境要求

```bash
pip install -r requirements.txt
```

主要依赖：
- Python 3.7+
- PyTorch 1.9+
- scapy (PCAP处理)
- scikit-learn
- tqdm
- matplotlib

## 项目文件结构

```
PERT/
├── data_processing.py      # 数据处理模块（PCAP解析、Bigram标记化）
├── model.py                # PERT模型架构
├── train_pretrain.py       # 掩码语言模型预训练脚本
├── train_classifier.py     # 流分类微调脚本
├── inference.py            # 推理工具
├── verify_code.py          # 代码完整性验证
├── requirements.txt        # Python依赖
├── README.md              # 本文件
├── DATA_PREPARATION.md    # ISCX数据集准备指南
└── QUICKSTART.md          # 快速入门
```

## 论文参数对照

### 模型架构参数

| 参数 | 论文值 | 说明 |
|------|--------|------|
| d_model | 256 | 模型维度 |
| n_layers | 6 | Transformer层数 |
| n_heads | 8 | 注意力头数 |
| d_ff | 1024 | 前馈网络维度 |
| max_seq_len | 128 | 最大序列长度 |
| max_packets | 3 | 每个流的数据包数 |
| mask_prob | 0.15 | MLM掩码概率 |
| vocab_size | 65539 | 词汇表大小(65536+3) |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| pre-train epochs | 50 | 预训练轮数 |
| classifier epochs | 30 | 分类训练轮数 |
| batch_size | 32 | 批次大小 |
| learning_rate | 5e-4 (pre-train) / 1e-4 (classifier) | 学习率 |
| dropout | 0.1 | Dropout率 |

## 使用方法

### 1. 准备数据

请按照 [DATA_PREPARATION.md](./DATA_PREPARATION.md) 下载并准备ISCX VPN-nonVPN数据集。

数据集包含12个类别：
- chat, email, file_transfer, p2p, streaming, voip
- vpn_chat, vpn_email, vpn_file_transfer, vpn_p2p, vpn_streaming, vpn_voip

### 2. （可选）预训练MLM

```bash
python train_pretrain.py \
    --data_dir /path/to/iscx_pcaps \
    --output_dir ./pretrain_checkpoints \
    --processed_data ./payloads_processed.pkl \
    --num_epochs 50
```

### 3. 训练分类器

直接从分类开始（可以使用或不使用预训练权重）：

```bash
# 从头开始训练
python train_classifier.py \
    --data_dir /path/to/iscx_dataset \
    --output_dir ./classifier_checkpoints \
    --processed_data ./flows_processed.pkl \
    --num_epochs 30

# 或使用预训练权重
python train_classifier.py \
    --data_dir /path/to/iscx_dataset \
    --pretrained_path ./pretrain_checkpoints/pert_pretrain_final.pt \
    --output_dir ./classifier_checkpoints \
    --num_epochs 30
```

### 4. 推理

```python
from inference import PERTClassifier

classifier = PERTClassifier(
    checkpoint_path="./classifier_checkpoints/classifier_best.pt",
    num_classes=12,
    max_packets=3
)

# 对流量进行预测
flow_payloads = [packet1_bytes, packet2_bytes, packet3_bytes]
pred_class, pred_probs = classifier.predict(flow_payloads)
```

## 验证代码

在开始之前，先验证代码完整性：

```bash
python verify_code.py
```

## 输出结果

训练过程会生成：
- 模型检查点
- 训练/验证损失曲线
- 分类报告（精确率、召回率、F1值）
- 混淆矩阵

## 完整流程示例

```bash
# 1. 确保数据集已准备
ls /path/to/iscx_dataset/
# chat/  email/  file_transfer/  ...

# 2. 验证代码
python verify_code.py

# 3. 预训练（可选但推荐）
python train_pretrain.py --data_dir /path/to/iscx_dataset --output_dir ./pretrain

# 4. 微调分类器
python train_classifier.py \
    --data_dir /path/to/iscx_dataset \
    --pretrained_path ./pretrain/pert_pretrain_final.pt \
    --output_dir ./classifier
```

## 论文引用

如果使用此代码，请引用原论文：

```bibtex
@inproceedings{he2020pert,
  title={PERT: Payload Encoding Representation from Transformer for Encrypted Traffic Classification},
  author={He, Hong Ye and Yang, Z. G. and Chen, X. N.},
  booktitle={2020 ITU Kaleidoscope: Industry-Driven Digital Transformation (ITU K)},
  year={2020}
}
```

同时请引用ISCX数据集：

```bibtex
@inproceedings{draper2016characterization,
  title={Characterization of Encrypted and VPN Traffic using Time-related Features},
  author={Draper-Gil, Gerard and Lashkari, Arash Habibi and Mamun, Mohammad Saiful Islam and Ghorbani, Ali A},
  booktitle={International Conference on Information Systems Security and Privacy},
  pages={407--414},
  year={2016}
}
```

## 常见问题

### Q: 没有数据集可以测试吗？
A: 本实现严格要求使用真实的ISCX数据集，没有内置合成数据。请按照DATA_PREPARATION.md下载。

### Q: 显存不足怎么办？
A: 减小batch_size或使用更小的模型：
```bash
python train_classifier.py --batch_size 16 --d_model 128 --n_layers 4
```

### Q: 可以使用自己的PCAP数据吗？
A: 可以！只要按照类别组织文件夹即可。

## 技术细节

### 模型架构图

```
输入数据包 → Bigram标记化 → [CLS] + tokens + [PAD]
                  ↓
         Token Embedding + Positional Encoding
                  ↓
         Transformer Encoder (6层)
                  ↓
         <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> Token Embedding
                  ↓
         （每个数据包重复以上步骤，取前3个）
                  ↓
         Concat: emb1 ⊕ emb2 ⊕ emb3
                  ↓
         Linear Layer → 12类输出
```

### 预训练目标

掩码语言模型（MLM）：随机掩码15%的token，预测被掩码的原始bigram。

### 分类方法

- 每个流取前3个数据包
- 每个数据包独立编码，取<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token
- 拼接3个<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token
- 通过单层线性层分类

## 许可证

本代码仅供研究使用。
