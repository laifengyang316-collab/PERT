# PERT Quick Start Guide

## Paper Summary

**PERT** (Payload Encoding Representation from Transformer) is a novel method for encrypted traffic classification that uses Transformer-based language modeling:

1. **Bigram Tokenization**: Byte pairs are treated as tokens (65536 possible values)
2. **Masked Language Model Pre-training**: Trains on unlabeled packets to learn contextual representations
3. **Flow-Level Classification**: Uses <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> tokens from multiple packets, concatenates them, and classifies

## What We've Implemented

### Core Files

| File | Description |
|------|-------------|
| `data_processing.py` | PCAP parsing, bigram tokenization, dataset classes |
| `model.py` | Transformer encoder, MLM head, classification head |
| `train_pretrain.py` | Pre-training script for masked language modeling |
| `train_classifier.py` | Fine-tuning script for traffic classification |
| `inference.py` | Model inference utilities |
| `test_model.py` | Quick test script to verify everything works |

### Quick Start

#### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2. Test the Installation

```bash
python test_model.py
```

#### 3. Train a Classification Model (from scratch)

```bash
python train_classifier.py \
    --num_classes 5 \
    --num_epochs 30 \
    --batch_size 32
```

#### 4. (Optional) Pre-train with MLM first

```bash
python train_pretrain.py \
    --num_epochs 50 \
    --batch_size 32

# Then fine-tune using the checkpoint
python train_classifier.py \
    --pretrained_path ./checkpoints/pert_pretrain_final.pt \
    --num_classes 5
```

## Using Your Own Data

### Preparing PCAP Files

1. Place your PCAP files in a directory
2. Use `data_processing.load_pcaps_from_dir()` to load them
3. Or modify the training scripts to use your data

### Custom Classification Dataset

```python
from data_processing import FlowClassificationDataset, create_classification_data

# Your data should be a list of (flow_payloads, label)
# flow_payloads is a list of bytes objects (packet payloads)
your_data = [
    ([b'packet1', b'packet2', b'packet3'], 0),
    ([b'packet1', b'packet2', b'packet3'], 1),
    # ... more samples
]

dataset = FlowClassificationDataset(your_data)
```

## Model Architecture Details

```
Input: [CLS] bigram1 bigram2 ... [PAD]
         ↓
    Token Embedding + Positional Encoding
         ↓
    [Transformer Encoder] × N_layers
         ↓
    [CLS] Embedding (per packet)
         ↓
    Concatenate (max_packets × d_model)
         ↓
    Classifier Head → Class Probabilities
```

## Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_layers | 6 |
| n_heads | 8 |
| d_ff | 1024 |
| max_seq_len | 128 |
| max_packets | 3 |
| vocab_size | 65539 (65536 + 3 special tokens) |

## Citation

```bibtex
@inproceedings{he2020pert,
  title={PERT: Payload Encoding Representation from Transformer for Encrypted Traffic Classification},
  author={He, Hong Ye and Yang, Z. G. and Chen, X. N.},
  booktitle={2020 ITU Kaleidoscope: Industry-Driven Digital Transformation (ITU K)},
  year={2020}
}
```
