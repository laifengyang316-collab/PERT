#!/usr/bin/env python3
"""
Simple test script to verify the PERT model implementation works correctly.
"""

import sys
import torch

print("=" * 60)
print("PERT Model Test")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    from data_processing import (
        bytes_to_bigrams,
        bigrams_to_bytes,
        generate_synthetic_data,
        create_classification_data,
        MaskedLMDataset,
        FlowClassificationDataset
    )
    from model import (
        PERTForMaskedLM,
        PERTForFlowClassification,
        count_parameters
    )
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Imports failed: {e}")
    sys.exit(1)

# Test 2: Test data processing
print("\n2. Testing data processing...")
try:
    # Test bigram conversion
    test_data = b"Hello, World! This is a test payload for PERT model."
    bigrams = bytes_to_bigrams(test_data)
    print(f"   ✓ Bigram conversion: {len(bigrams)} bigrams created")
    
    # Test synthetic data generation
    payloads = generate_synthetic_data(100)
    print(f"   ✓ Synthetic data generation: {len(payloads)} payloads")
    
    # Test dataset creation
    mlm_dataset = MaskedLMDataset(payloads)
    sample = mlm_dataset[0]
    print(f"   ✓ MLM Dataset created: masked_ids={sample[0].shape}")
    
    # Test classification data
    class_data = create_classification_data(num_classes=5, num_flows_per_class=20)
    cls_dataset = FlowClassificationDataset(class_data)
    sample_flow, sample_label = cls_dataset[0]
    print(f"   ✓ Classification Dataset created: flow={sample_flow.shape}, label={sample_label}")
    
except Exception as e:
    print(f"   ✗ Data processing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test model creation
print("\n3. Testing model creation...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    # Test MLM model
    mlm_model = PERTForMaskedLM(
        vocab_size=65539,  # 65536 + 3 special tokens
        d_model=128,       # Smaller for testing
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_len=64,
        dropout=0.1
    )
    mlm_model = mlm_model.to(device)
    print(f"   ✓ MLM Model created: {count_parameters(mlm_model):,} parameters")
    
    # Test forward pass of MLM
    batch_size = 2
    seq_len = 64
    dummy_input = torch.randint(0, 65539, (batch_size, seq_len)).to(device)
    logits = mlm_model(dummy_input)
    print(f"   ✓ MLM forward pass: output shape={logits.shape}")
    
    # Test classification model
    cls_model = PERTForFlowClassification(
        num_classes=5,
        vocab_size=65539,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        max_len=64,
        max_packets=3,
        dropout=0.1
    )
    cls_model = cls_model.to(device)
    print(f"   ✓ Classification Model created: {count_parameters(cls_model):,} parameters")
    
    # Test forward pass of classification model
    dummy_flow_input = torch.randint(0, 65539, (batch_size, 3, seq_len)).to(device)
    cls_logits = cls_model(dummy_flow_input)
    print(f"   ✓ Classification forward pass: output shape={cls_logits.shape}")
    
except Exception as e:
    print(f"   ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test training loop components
print("\n4. Testing training components...")
try:
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    # Test small training step
    small_payloads = generate_synthetic_data(50)
    small_dataset = MaskedLMDataset(small_payloads, max_seq_len=32)
    small_loader = DataLoader(small_dataset, batch_size=4, shuffle=True)
    
    small_model = PERTForMaskedLM(
        vocab_size=65539,
        d_model=64,
        n_layers=1,
        n_heads=2,
        d_ff=256,
        max_len=32
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(small_model.parameters(), lr=1e-4)
    
    # One training step
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
    
    print(f"   ✓ Small training step successful: loss={loss.item():.4f}")
    
except Exception as e:
    print(f"   ✗ Training components failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nNext steps:")
print("1. Run 'python train_classifier.py' to train a classification model")
print("2. Run 'python train_pretrain.py' for MLM pre-training (optional)")
print("3. Check README.md for detailed usage instructions")
