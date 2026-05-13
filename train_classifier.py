import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import PERTForFlowClassification, count_parameters
from data_processing import (
    FlowClassificationDataset,
    load_iscx_dataset,
    save_processed_data,
    load_processed_data,
    TOTAL_VOCAB_SIZE
)


def parse_args():
    parser = argparse.ArgumentParser(description="PERT Flow Classification (ISCX Dataset - 12 Classes)")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing ISCX dataset (PCAP files organized by class)")
    parser.add_argument("--processed_data", type=str, default=None,
                        help="Path to save/load processed data")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained MLM checkpoint (optional but recommended)")
    parser.add_argument("--output_dir", type=str, default="./classifier_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--num_classes", type=int, default=12,
                        help="Number of classes (default: 12 for ISCX)")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Maximum sequence length per packet (default: 128)")
    parser.add_argument("--max_packets", type=int, default=3,
                        help="Maximum number of packets per flow (default: 3)")
    parser.add_argument("--max_flows_per_class", type=int, default=None,
                        help="Maximum number of flows per class to load (optional)")
    parser.add_argument("--use_full_dataset", action="store_true",
                        help="Use full dataset for training (no train/validation split, for final model)")
    
    parser.add_argument("--d_model", type=int, default=256,
                        help="Model dimension (default: 256)")
    parser.add_argument("--n_layers", type=int, default=6,
                        help="Number of transformer layers (default: 6)")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of attention heads (default: 8)")
    parser.add_argument("--d_ff", type=int, default=1024,
                        help="Feed-forward dimension (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights during training")
    parser.add_argument("--save_interval", type=int, default=5,
                        help="Save interval (epochs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch + 1}/{num_epochs}", leave=False)
    
    for batch_idx, (input_ids, labels) in enumerate(progress_bar):
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        for input_ids, labels in progress_bar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            logits = model(input_ids)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    args = parse_args()
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PERT Flow Classification Fine-tuning (ISCX Dataset - 12 Classes)")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Using pretrained model: {args.pretrained_path is not None}")
    print(f"Freeze encoder: {args.freeze_encoder}")
    print("-" * 80)
    print("Model parameters (per paper):")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  Packets per flow: {args.max_packets}")
    print(f"  Number of classes: {args.num_classes}")
    print("=" * 80)
    
    if args.processed_data and os.path.exists(args.processed_data):
        print(f"\nLoading processed data from {args.processed_data}...")
        flow_data, idx_to_class_name = load_processed_data(args.processed_data)
    else:
        if not os.path.exists(args.data_dir):
            print(f"\nError: Data directory not found: {args.data_dir}")
            print("Please follow DATA_PREPARATION.md to download ISCX dataset.")
            return
        
        print("\nLoading ISCX dataset...")
        flow_data, idx_to_class_name = load_iscx_dataset(
            args.data_dir,
            max_flows_per_class=args.max_flows_per_class,
            packets_per_flow=args.max_packets
        )
        
        if len(flow_data) == 0:
            print("\nError: No valid flow data found!")
            return
        
        if args.processed_data:
            save_processed_data((flow_data, idx_to_class_name), args.processed_data)
    
    num_classes = len(idx_to_class_name)
    print(f"\nDetected classes: {num_classes}")
    print("Class list:")
    for idx, name in idx_to_class_name.items():
        count = len([f for f in flow_data if f[1] == idx])
        print(f"  {idx}: {name} ({count} flows)")
    
    dataset = FlowClassificationDataset(
        flow_data=flow_data,
        max_packets_per_flow=args.max_packets,
        max_seq_len=args.max_seq_len
    )
    
    if args.use_full_dataset:
        print(f"\nUsing full dataset for training: {len(dataset)} samples")
        train_dataset = dataset
        val_dataset = None
    else:
        labels = [label for _, label in flow_data]
        indices = list(range(len(dataset)))
        
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            stratify=labels,
            random_state=42
        )
        
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        
        print(f"\nTraining set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
    
    print("\nCreating model...")
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"Loading pretrained encoder from {args.pretrained_path}...")
        model = PERTForFlowClassification.from_pretrained(
            args.pretrained_path,
            num_classes=num_classes,
            max_packets=args.max_packets,
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_len=args.max_seq_len,
            dropout=args.dropout
        )
    else:
        print("Training from scratch...")
        model = PERTForFlowClassification(
            num_classes=num_classes,
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            max_len=args.max_seq_len,
            max_packets=args.max_packets,
            dropout=args.dropout
        )
    
    if args.freeze_encoder:
        print("Freezing encoder weights...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    model = model.to(args.device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    class_map_path = os.path.join(args.output_dir, "class_names.pkl")
    save_processed_data(idx_to_class_name, class_map_path)
    
    all_labels = [label for _, label in flow_data]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
    print(f"\nClass weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    print("\nStarting training...")
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0.0
    
    for epoch in range(args.num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device, epoch, args.num_epochs)
        
        val_loss, val_acc = float('nan'), float('nan')
        val_preds, val_labels = [], []
        
        if val_loader is not None:
            val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, args.device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        if val_loader is not None:
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        if val_loader is not None:
            print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_save_path = os.path.join(args.output_dir, "classifier_best.pt")
                save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc, 'class_names': idx_to_class_name}, best_save_path)
                print(f"  -> New best validation accuracy!")
        
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.output_dir, f"classifier_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc if val_loader else None, 'class_names': idx_to_class_name}, save_path)
    
    final_save_path = os.path.join(args.output_dir, "classifier_final.pt")
    if val_loader is not None:
        save_checkpoint(model, optimizer, args.num_epochs, {'val_acc': val_accs[-1], 'class_names': idx_to_class_name}, final_save_path)
    else:
        save_checkpoint(model, optimizer, args.num_epochs, {'class_names': idx_to_class_name}, final_save_path)
    
    if val_loader is not None:
        print("\n" + "=" * 80)
        print("Final Evaluation Report")
        print("=" * 80)
        
        model.load_state_dict(torch.load(best_save_path)['model_state_dict'])
        _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, args.device)
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(final_labels, final_preds))
        print("\nClassification Report:")
        target_names = [idx_to_class_name[i] for i in range(len(idx_to_class_name))]
        print(classification_report(final_labels, final_preds, target_names=target_names, digits=4))
    
    if val_loader is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'training_metrics.png'))
        print(f"\nTraining curve saved to {os.path.join(args.output_dir, 'training_metrics.png')}")
    
    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    if val_loader is not None:
        print(f"Best model saved to: {best_save_path}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    else:
        print(f"Full dataset model saved to: {os.path.join(args.output_dir, 'classifier_final.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()