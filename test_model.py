#!/usr/bin/env python3
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("=" * 60)
print("PERT Model Test")
print("=" * 60)

print("\n1. Testing imports...")
try:
    from data_processing import (
        bytes_to_bigrams,
        bigrams_to_bytes,
        MaskedLMDataset,
        FlowClassificationDataset,
        TOTAL_VOCAB_SIZE,
        SPECIAL_TOKENS,
        CLS_TOKEN,
        MASK_TOKEN,
        PAD_TOKEN
    )
    from model import (
        PERTForMaskedLM,
        PERTForFlowClassification,
        count_parameters
    )
    print("   OK Imports successful")
except Exception as e:
    print(f"   FAIL Imports failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n2. Testing data processing...")
try:
    test_data = b"Hello, World! This is a test payload for PERT model."
    bigrams = bytes_to_bigrams(test_data)
    print(f"   OK Bigram conversion: {len(bigrams)} bigrams from {len(test_data)} bytes")

    recovered = bigrams_to_bytes(bigrams)
    expected = test_data[:len(test_data) - (len(test_data) % 2)]
    assert recovered == expected, f"Round-trip failed: {recovered} != {expected}"
    print(f"   OK Bigram round-trip conversion")

    import numpy as np
    dummy_payloads = [test_data for _ in range(16)]
    mlm_dataset = MaskedLMDataset(dummy_payloads, max_seq_len=32)
    sample = mlm_dataset[0]
    print(f"   OK MLM Dataset: masked_ids={sample[0].shape}")

    dummy_flow_data = [([b'packet1', b'packet2', b'packet3'], 0) for _ in range(8)]
    cls_dataset = FlowClassificationDataset(dummy_flow_data, max_packets_per_flow=3, max_seq_len=32)
    sample_flow, sample_label = cls_dataset[0]
    print(f"   OK Classification Dataset: flow={sample_flow.shape}, label={sample_label}")

except Exception as e:
    print(f"   FAIL Data processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n3. Testing model creation...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    mlm_model = PERTForMaskedLM(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_len=64,
        dropout=0.1
    )
    mlm_model = mlm_model.to(device)
    print(f"   OK MLM Model created: {count_parameters(mlm_model):,} parameters")

    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, TOTAL_VOCAB_SIZE, (batch_size, seq_len)).to(device)
    logits = mlm_model(dummy_input)
    print(f"   OK MLM forward pass: output shape={logits.shape}")

    cls_model = PERTForFlowClassification(
        num_classes=5,
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_len=64,
        max_packets=3,
        dropout=0.1
    )
    cls_model = cls_model.to(device)
    print(f"   OK Classification Model created: {count_parameters(cls_model):,} parameters")

    dummy_flow_input = torch.randint(0, TOTAL_VOCAB_SIZE, (batch_size, 3, seq_len)).to(device)
    cls_logits = cls_model(dummy_flow_input)
    print(f"   OK Classification forward pass: output shape={cls_logits.shape}")

except Exception as e:
    print(f"   FAIL Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Testing training components...")
try:
    np = __import__('numpy')
    dummy_payloads = [b'test payload data for training' for _ in range(32)]
    small_dataset = MaskedLMDataset(dummy_payloads, max_seq_len=32)
    small_loader = DataLoader(small_dataset, batch_size=4, shuffle=True)

    small_model = PERTForMaskedLM(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=64,
        n_layers=1,
        n_heads=2,
        d_ff=256,
        max_len=32
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(small_model.parameters(), lr=1e-4)

    small_model.train()
    masked_ids, true_ids, mask = next(iter(small_loader))
    masked_ids = masked_ids.to(device)
    true_ids = true_ids.to(device)
    mask = mask.to(device)

    optimizer.zero_grad()
    logits = small_model(masked_ids)

    active_loss = mask.view(-1) == 1
    active_logits = logits.view(-1, logits.size(-1))[active_loss]
    active_labels = true_ids.view(-1)[active_loss]
    loss = criterion(active_logits, active_labels)

    loss.backward()
    optimizer.step()

    print(f"   OK Training step successful: loss={loss.item():.4f}")

except Exception as e:
    print(f"   FAIL Training components failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. Testing ALBERT-specific features...")
try:
    from data_processing import TOTAL_VOCAB_SIZE
    from model import TransformerEncoder

    encoder = TransformerEncoder(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_len=64,
        embedding_size=128
    )

    assert hasattr(encoder, 'shared_layer'), "Missing shared_layer (cross-layer sharing)"
    assert hasattr(encoder, 'embedding_projection'), "Missing embedding_projection (factorized embedding)"
    assert encoder.embedding_size == 128, f"Expected embedding_size=128, got {encoder.embedding_size}"

    dummy_input = torch.randint(0, TOTAL_VOCAB_SIZE, (2, 64))
    output = encoder(dummy_input)
    assert output.shape == (2, 64, 128), f"Unexpected output shape: {output.shape}"
    print(f"   OK ALBERT architecture verified: cross-layer sharing + factorized embedding")

except Exception as e:
    print(f"   FAIL ALBERT verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nNext steps:")
print("1. python train_pretrain.py --data_dir /path/to/iscx_dataset")
print("2. python train_classifier.py --data_dir /path/to/iscx_dataset --pretrained_path ./pretrain_checkpoints/pert_pretrain_final.pt")