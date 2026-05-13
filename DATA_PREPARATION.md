# ISCX VPN-nonVPN Dataset Preparation

This guide explains how to obtain and prepare the ISCX VPN-nonVPN dataset for training the PERT model.

## About the Dataset

The ISCX VPN-nonVPN dataset is a collection of encrypted network traffic captures used for research in traffic classification. It contains:
- Regular traffic (non-VPN)
- VPN traffic
- Various application types: browsing, chat, email, file transfer, streaming, VoIP, P2P

## Downloading the Dataset

### Official Source
Visit the official UNB website:
https://www.unb.ca/cic/datasets/vpn.html

### Alternative Sources
If the official site is unavailable, try these mirrors:
- Kaggle (search for "ISCX VPN-nonVPN")
- Academic torrents
- Research data repositories

## Dataset Structure

After downloading, you should have PCAP files organized in one of these ways:

### Option 1: Class-based Subdirectories (Recommended)
```
iscx_dataset/
├── browsing/
│   ├── browse1.pcap
│   ├── browse2.pcap
│   └── ...
├── chat/
│   ├── chat1.pcap
│   └── ...
├── email/
├── file_transfer/
├── streaming/
├── voip/
├── p2p/
├── vpn_browsing/
├── vpn_chat/
└── ...
```

### Option 2: All PCAP Files in One Directory
```
iscx_dataset/
├── browsing1.pcap
├── browsing2.pcap
├── chat1.pcap
├── vpn_browsing1.pcap
└── ...
```

The code will automatically detect the structure. If files are in one directory, it will try to infer class names from filenames.

## Recommended Classes

The paper uses 12 classes. Here's the list:

| Class Name | Description |
|------------|-------------|
| browsing | Web browsing traffic |
| chat | Instant messaging traffic |
| email | Email traffic |
| file_transfer | File transfer (FTP, etc.) |
| streaming | Video/audio streaming |
| voip | Voice over IP |
| p2p | Peer-to-peer file sharing |
| vpn_browsing | VPN web browsing |
| vpn_chat | VPN chat |
| vpn_email | VPN email |
| vpn_file_transfer | VPN file transfer |
| vpn_streaming | VPN streaming |
| vpn_voip | VPN VoIP |
| vpn_p2p | VPN P2P |

## Preprocessing the Dataset

### Step 1: Organize Files
For best results, organize your PCAP files into class-named subdirectories as shown in Option 1 above.

### Step 2: Verify PCAP Files
Make sure your PCAP files are valid and contain network traffic with payloads (not just TCP handshakes).

### Step 3: (Optional) Filter Traffic
If needed, you can filter the PCAP files to only include certain types of traffic:
- TCP only
- Only flows with payload data
- Remove known malicious traffic

## Using the Dataset with Our Code

### For Pre-training (MLM)
Use any collection of PCAP files (unlabeled data works best):

```bash
python train_pretrain.py \
    --data_dir ./iscx_dataset \
    --output_dir ./checkpoints \
    --processed_data ./processed_payloads.pkl \
    --num_epochs 50
```

### For Classification
Use the organized dataset:

```bash
python train_classifier.py \
    --data_dir ./iscx_dataset \
    --output_dir ./classifier_checkpoints \
    --processed_data ./processed_flows.pkl \
    --pretrained_path ./checkpoints/pert_pretrain_final.pt \
    --num_epochs 30
```

## Processing Options

### Limiting Data Size
If the dataset is too large, use these options:

```bash
# Limit flows per class
python train_classifier.py \
    --data_dir ./iscx_dataset \
    --max_flows_per_class 1000
```

### Caching Processed Data
Save time by caching processed data:

```bash
# First run - saves processed data
python train_classifier.py \
    --data_dir ./iscx_dataset \
    --processed_data ./iscx_processed.pkl

# Subsequent runs - loads from cache
python train_classifier.py \
    --data_dir ./iscx_dataset \
    --processed_data ./iscx_processed.pkl
```

## Expected Dataset Size

The full ISCX VPN-nonVPN dataset contains:
- Multiple GB of PCAP files
- Millions of packets
- Thousands of flows

For training, we recommend:
- At least 10,000 packets for pre-training
- At least 1,000 flows per class for classification

## Troubleshooting

### No Payloads Found
- Check if your PCAP files contain actual data (not just SYN/ACK)
- Try using Wireshark to inspect the PCAP files
- Some VPN traffic might be fully encrypted with no readable payload

### Out of Memory Errors
- Reduce batch size (`--batch_size`)
- Limit data with `--max_flows_per_class`
- Use a smaller model (`--d_model 128 --n_layers 4`)

### Class Imbalance
- The dataset might have uneven class distribution
- Consider oversampling minority classes
- Or use class weights in the loss function

## Using Custom Data

If you want to use your own PCAP files:

1. Organize them into class-named subdirectories
2. Or name files to indicate class (e.g., `myapp_traffic1.pcap`)
3. The code will work with any valid PCAP files

## Citation

If you use this dataset, please cite:

```
@inproceedings{draper2016characterization,
  title={Characterization of Encrypted and VPN Traffic using Time-related Features},
  author={Draper-Gil, Gerard and Lashkari, Arash Habibi and Mamun, Mohammad Saiful Islam and Ghorbani, Ali A},
  booktitle={International Conference on Information Systems Security and Privacy},
  pages={407--414},
  year={2016}
}
```

And of course, cite the original PERT paper!
