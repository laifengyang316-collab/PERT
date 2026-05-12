import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from model import PERTForMaskedLM, count_parameters
from data_processing import (
    MaskedLMDataset,
    load_pcaps_from_dir,
    save_processed_data,
    load_processed_data,
    TOTAL_VOCAB_SIZE
)


def parse_args():
    parser = argparse.ArgumentParser(description="PERT Masked Language Model Pre-training (ISCX Dataset)")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing PCAP files (ISCX Dataset)")
    parser.add_argument("--output_dir", type=str, default="./pretrain_checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--processed_data", type=str, default=None,
                        help="Path to save/load processed data")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate (default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Maximum sequence length (default: 128)")
    parser.add_argument("--mask_prob", type=float, default=0.15,
                        help="Masking probability for MLM (default: 0.15)")
    parser.add_argument("--max_packets_per_file", type=int, default=None,
                        help="Maximum number of packets per PCAP file (default: None, load all)")
    
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
    
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save interval (epochs)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}", leave=False)
    
    for batch_idx, (masked_ids, true_ids, mask) in enumerate(progress_bar):
        masked_ids = masked_ids.to(device)
        true_ids = true_ids.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        logits = model(masked_ids)
        
        active_loss = mask.view(-1) == 1
        active_logits = logits.view(-1, logits.size(-1))[active_loss]
        active_labels = true_ids.view(-1)[active_loss]
        loss = criterion(active_logits, active_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_path)
    print(f"Checkpoint saved to {save_path}")


def main():
    global args
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("PERT 掩码语言模型预训练")
    print("=" * 80)
    print(f"设备: {args.device}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据目录: {args.data_dir}")
    print("-" * 80)
    print("模型参数 (按论文设置):")
    print(f"  d_model: {args.d_model}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  词汇表大小: {TOTAL_VOCAB_SIZE}")
    print("=" * 80)
    
    # 加载数据
    if args.processed_data and os.path.exists(args.processed_data):
        print(f"Loading processed data from {args.processed_data}...")
        payloads = load_processed_data(args.processed_data)
    else:
        if not os.path.exists(args.data_dir):
            print(f"\n错误: 数据目录不存在: {args.data_dir}")
            print("请按照 DATA_PREPARATION.md 下载并准备ISCX数据集。")
            return
        
        print(f"\n正在从 {args.data_dir} 加载数据...")
        if args.max_packets_per_file:
            print(f"每个PCAP文件最多加载 {args.max_packets_per_file} 个数据包")
        payloads = load_pcaps_from_dir(args.data_dir, max_packets_per_file=args.max_packets_per_file)
        
        if len(payloads) == 0:
            print("\n错误: 没有找到有效的数据包！")
            print("请检查PCAP文件是否包含有效的流量数据。")
            return
        
        print(f"成功加载 {len(payloads)} 个数据包")
        
        if args.processed_data:
            save_processed_data(payloads, args.processed_data)
    
    # 创建数据集
    dataset = MaskedLMDataset(
        payloads=payloads,
        max_seq_len=args.max_seq_len,
        mask_prob=args.mask_prob
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 创建模型
    print("\n正在创建模型...")
    model = PERTForMaskedLM(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_len=args.max_seq_len,
        dropout=args.dropout
    )
    
    model = model.to(args.device)
    print(f"模型参数数量: {count_parameters(model):,}")
    
    # 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # 开始训练
    print("\n开始预训练...")
    train_losses = []
    
    for epoch in range(args.num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, args.device, epoch)
        train_losses.append(avg_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch + 1}/{args.num_epochs} - 平均损失: {avg_loss:.4f}")
        
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.num_epochs:
            save_path = os.path.join(args.output_dir, f"pert_pretrain_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, save_path)
    
    # 保存最终模型
    final_save_path = os.path.join(args.output_dir, "pert_pretrain_final.pt")
    save_checkpoint(model, optimizer, args.num_epochs, train_losses[-1], final_save_path)
    
    # 保存训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PERT Pre-training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    print(f"\n训练曲线已保存至 {os.path.join(args.output_dir, 'training_loss.png')}")
    
    print("\n" + "=" * 80)
    print("预训练完成！")
    print(f"最终模型保存至: {final_save_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
