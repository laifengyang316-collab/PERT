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
    
    # 论文参数
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
    
    # 固定随机种子（确保可复现性）
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PERT 流分类微调 (ISCX数据集 - 12类)")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据目录: {args.data_dir}")
    print(f"使用预训练模型: {args.pretrained_path is not None}")
    print(f"冻结编码器: {args.freeze_encoder}")
    print("-" * 80)
    print("模型参数 (按论文设置):")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  每个流数据包数: {args.max_packets}")
    print(f"  类别数: {args.num_classes}")
    print("=" * 80)
    
    # 加载数据
    if args.processed_data and os.path.exists(args.processed_data):
        print(f"\n正在从 {args.processed_data} 加载处理后的数据...")
        flow_data, idx_to_class_name = load_processed_data(args.processed_data)
    else:
        if not os.path.exists(args.data_dir):
            print(f"\n错误: 数据目录不存在: {args.data_dir}")
            print("请按照 DATA_PREPARATION.md 下载并准备ISCX数据集。")
            return
        
        print("\n正在加载ISCX数据集...")
        flow_data, idx_to_class_name = load_iscx_dataset(
            args.data_dir,
            max_flows_per_class=args.max_flows_per_class,
            packets_per_flow=args.max_packets
        )
        
        if len(flow_data) == 0:
            print("\n错误: 没有找到有效的流数据！")
            return
        
        if args.processed_data:
            save_processed_data((flow_data, idx_to_class_name), args.processed_data)
    
    num_classes = len(idx_to_class_name)
    print(f"\n检测到的类别数: {num_classes}")
    print("类别列表:")
    for idx, name in idx_to_class_name.items():
        count = len([f for f in flow_data if f[1] == idx])
        print(f"  {idx}: {name} ({count} flows)")
    
    # 创建数据集
    dataset = FlowClassificationDataset(
        flow_data=flow_data,
        max_packets_per_flow=args.max_packets,
        max_seq_len=args.max_seq_len
    )
    
    if args.use_full_dataset:
        print(f"\n使用完整数据集训练: {len(dataset)} 样本")
        train_dataset = dataset
        val_dataset = None
    else:
        # 划分训练集和验证集 (80/20) - 使用stratified split确保类别分布一致
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
        
        print(f"\n训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
    
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
    
    # 创建模型
    print("\n正在创建模型...")
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"正在从 {args.pretrained_path} 加载预训练编码器...")
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
        print("从头开始训练...")
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
    
    # 冻结编码器（如果指定）
    if args.freeze_encoder:
        print("冻结编码器权重...")
        for param in model.encoder.parameters():
            param.requires_grad = False
    
    model = model.to(args.device)
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 保存类别映射
    class_map_path = os.path.join(args.output_dir, "class_names.pkl")
    save_processed_data(idx_to_class_name, class_map_path)
    
    # 计算class weight（处理类别不平衡）
    all_labels = [label for _, label in flow_data]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)
    print(f"\n类别权重: {class_weights}")
    
    # 训练设置 - 使用class weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 开始训练
    print("\n开始训练...")
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
        print(f"  训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        if val_loader is not None:
            print(f"  验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_save_path = os.path.join(args.output_dir, "classifier_best.pt")
                save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc, 'class_names': idx_to_class_name}, best_save_path)
                print(f"  -> 新的最佳验证准确率！")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.output_dir, f"classifier_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc if val_loader else None, 'class_names': idx_to_class_name}, save_path)
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "classifier_final.pt")
    if val_loader is not None:
        save_checkpoint(model, optimizer, args.num_epochs, {'val_acc': val_accs[-1], 'class_names': idx_to_class_name}, final_save_path)
    else:
        save_checkpoint(model, optimizer, args.num_epochs, {'class_names': idx_to_class_name}, final_save_path)
    
    # 最终评估报告
    if val_loader is not None:
        print("\n" + "=" * 80)
        print("最终评估报告")
        print("=" * 80)
        
        model.load_state_dict(torch.load(best_save_path)['model_state_dict'])
        _, _, final_preds, final_labels = evaluate(model, val_loader, criterion, args.device)
        
        print("\n混淆矩阵:")
        print(confusion_matrix(final_labels, final_preds))
        print("\n分类报告:")
        target_names = [idx_to_class_name[i] for i in range(len(idx_to_class_name))]
        print(classification_report(final_labels, final_preds, target_names=target_names, digits=4))
    
    # 保存训练曲线
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
        print(f"\n训练曲线已保存至 {os.path.join(args.output_dir, 'training_metrics.png')}")
    
    print("\n" + "=" * 80)
    print("微调完成！")
    if val_loader is not None:
        print(f"最佳模型保存至: {best_save_path}")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
    else:
        print(f"完整数据集模型保存至: {os.path.join(args.output_dir, 'classifier_final.pt')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
