import os
import struct
import random
import pickle
import numpy as np
from scapy.all import rdpcap, IP, TCP, UDP
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

CLS_TOKEN = 0
MASK_TOKEN = 1
PAD_TOKEN = 2

VOCAB_SIZE = 65536
SPECIAL_TOKENS = 3
TOTAL_VOCAB_SIZE = VOCAB_SIZE + SPECIAL_TOKENS

ISCX_CLASSES = [
    'chat', 'email', 'file_transfer', 'p2p',
    'streaming', 'voip', 'vpn_chat', 'vpn_email',
    'vpn_file_transfer', 'vpn_p2p', 'vpn_streaming', 'vpn_voip'
]


def bytes_to_bigrams(byte_data: bytes) -> List[int]:
    bigrams = []
    for i in range(0, len(byte_data) - 1, 2):
        if i + 1 < len(byte_data):
            bigram = (byte_data[i] << 8) | byte_data[i + 1]
            bigrams.append(bigram + SPECIAL_TOKENS)
    return bigrams


def bigrams_to_bytes(bigrams: List[int]) -> bytes:
    byte_data = []
    for bg in bigrams:
        if bg >= SPECIAL_TOKENS:
            bg -= SPECIAL_TOKENS
            byte1 = (bg >> 8) & 0xFF
            byte2 = bg & 0xFF
            byte_data.append(byte1)
            byte_data.append(byte2)
    return bytes(byte_data)


def extract_payload(packet) -> bytes:
    payload = b''
    if IP in packet:
        if TCP in packet:
            payload = bytes(packet[TCP].payload)
        elif UDP in packet:
            payload = bytes(packet[UDP].payload)
    return payload


def get_flow_key(packet) -> Optional[Tuple]:
    if IP not in packet:
        return None
    
    ip_layer = packet[IP]
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    
    if TCP in packet:
        trans_layer = packet[TCP]
        proto = 'TCP'
    elif UDP in packet:
        trans_layer = packet[UDP]
        proto = 'UDP'
    else:
        return None
    
    src_port = trans_layer.sport
    dst_port = trans_layer.dport
    
    if src_ip < dst_ip:
        return (src_ip, src_port, dst_ip, dst_port, proto)
    elif src_ip > dst_ip:
        return (dst_ip, dst_port, src_ip, src_port, proto)
    else:
        if src_port <= dst_port:
            return (src_ip, src_port, dst_ip, dst_port, proto)
        else:
            return (dst_ip, dst_port, src_ip, src_port, proto)


def parse_pcap_flows(pcap_path: str, max_flows: int = None,
                     packets_per_flow: int = 3) -> Dict[Tuple, List[bytes]]:
    flows = defaultdict(list)
    
    try:
        packets = rdpcap(pcap_path)
        for packet in packets:
            flow_key = get_flow_key(packet)
            if flow_key is None:
                continue
            
            payload = extract_payload(packet)
            if len(payload) >= 4:
                flows[flow_key].append(payload)
            
            if max_flows and len(flows) >= max_flows:
                break
                
    except Exception as e:
        print(f"Error parsing {pcap_path}: {e}")
    
    for flow_key in flows:
        flows[flow_key] = flows[flow_key][:packets_per_flow]
    
    return dict(flows)


def parse_pcap(pcap_path: str, max_packets: int = None) -> List[bytes]:
    payloads = []
    try:
        packets = rdpcap(pcap_path)
        total = min(max_packets, len(packets)) if max_packets else len(packets)
        from tqdm import tqdm
        
        for i, packet in enumerate(tqdm(packets, desc=f"  Parsing {os.path.basename(pcap_path)}", leave=False)):
            if max_packets is not None and i >= max_packets:
                break
            payload = extract_payload(packet)
            if len(payload) >= 4:
                payloads.append(payload)
    except Exception as e:
        print(f"Error parsing {pcap_path}: {e}")
    return payloads


def load_pcaps_from_dir(directory: str, max_packets_per_file: int = None) -> List[bytes]:
    all_payloads = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.pcap') or filename.endswith('.pcapng'):
                filepath = os.path.join(root, filename)
                print(f"Loading {filepath}...")
                payloads = parse_pcap(filepath, max_packets_per_file)
                all_payloads.extend(payloads)
    return all_payloads


def load_iscx_dataset(dataset_dir: str, max_flows_per_class: int = None,
                      packets_per_flow: int = 3) -> Tuple[List[Tuple[List[bytes], int]], Dict[int, str]]:
    flow_data = []
    class_name_to_idx = {}
    idx_to_class_name = {}
    
    print(f"Loading ISCX dataset from {dataset_dir}...")
    
    for root, dirs, files in os.walk(dataset_dir):
        for filename in files:
            if not (filename.endswith('.pcap') or filename.endswith('.pcapng')):
                continue
            
            class_name = None
            filename_lower = filename.lower()
            
            is_vpn = 'vpn' in filename_lower or 'VPN' in root
            
            for candidate in ISCX_CLASSES:
                if candidate in filename_lower:
                    class_name = candidate
                    break
            
            if class_name is None:
                folder_name = os.path.basename(root).lower()
                for candidate in ISCX_CLASSES:
                    if candidate in folder_name:
                        class_name = candidate
                        break
            
            if class_name is None:
                if 'bittorrent' in filename_lower or 'torrent' in filename_lower:
                    class_name = 'vpn_p2p' if is_vpn else 'p2p'
                elif 'ftps' in filename_lower or 'sftp' in filename_lower or 'scp' in filename_lower:
                    class_name = 'vpn_file_transfer' if is_vpn else 'file_transfer'
                elif 'email' in filename_lower:
                    class_name = 'vpn_email' if is_vpn else 'email'
                elif any(x in filename_lower for x in ['video', 'youtube', 'netflix', 'vimeo', 'spotify']):
                    class_name = 'vpn_streaming' if is_vpn else 'streaming'
                elif any(x in filename_lower for x in ['audio', 'voip', 'voice']):
                    class_name = 'vpn_voip' if is_vpn else 'voip'
                elif any(x in filename_lower for x in ['chat', 'aim', 'icq', 'facebook', 'gmail', 'hangouts', 'skype']):
                    if 'skype_file' in filename_lower:
                        class_name = 'vpn_file_transfer' if is_vpn else 'file_transfer'
                    else:
                        class_name = 'vpn_chat' if is_vpn else 'chat'
            
            if class_name is None:
                print(f"Skipping (unknown class): {filename}")
                continue
            
            if class_name not in class_name_to_idx:
                class_idx = len(class_name_to_idx)
                class_name_to_idx[class_name] = class_idx
                idx_to_class_name[class_idx] = class_name
            
            filepath = os.path.join(root, filename)
            print(f"Processing {filepath} as class: {class_name}")
            
            flows = parse_pcap_flows(filepath, packets_per_flow=packets_per_flow)
            
            for flow_payloads in flows.values():
                if len(flow_payloads) > 0:
                    flow_data.append((flow_payloads, class_name_to_idx[class_name]))
                    
                    if max_flows_per_class:
                        class_count = len([f for f in flow_data if f[1] == class_name_to_idx[class_name]])
                        if class_count >= max_flows_per_class:
                            break
    
    random.shuffle(flow_data)
    
    print(f"Loaded {len(flow_data)} flows from {len(class_name_to_idx)} classes")
    for idx, name in idx_to_class_name.items():
        count = len([f for f in flow_data if f[1] == idx])
        print(f"  Class {idx} ({name}): {count} flows")
    
    return flow_data, idx_to_class_name


def save_processed_data(data, filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filepath}")


def load_processed_data(filepath: str):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {filepath}")
    return data


class PacketDataset:
    def __init__(self, payloads: List[bytes], max_seq_len: int = 128):
        self.payloads = payloads
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.payloads)
    
    def __getitem__(self, idx):
        payload = self.payloads[idx]
        bigrams = bytes_to_bigrams(payload)
        
        if len(bigrams) > self.max_seq_len - 1:
            bigrams = bigrams[:self.max_seq_len - 1]
        
        tokens = [CLS_TOKEN] + bigrams
        if len(tokens) < self.max_seq_len:
            tokens += [PAD_TOKEN] * (self.max_seq_len - len(tokens))
        
        return np.array(tokens, dtype=np.int64)


class MaskedLMDataset:
    def __init__(self, payloads: List[bytes], max_seq_len: int = 128, mask_prob: float = 0.15):
        self.payloads = payloads
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.payloads)
    
    def __getitem__(self, idx):
        payload = self.payloads[idx]
        bigrams = bytes_to_bigrams(payload)
        
        if len(bigrams) > self.max_seq_len - 1:
            bigrams = bigrams[:self.max_seq_len - 1]
        
        tokens = [CLS_TOKEN] + bigrams
        if len(tokens) < self.max_seq_len:
            tokens += [PAD_TOKEN] * (self.max_seq_len - len(tokens))
        
        tokens = np.array(tokens, dtype=np.int64)
        masked_tokens = tokens.copy()
        mask = np.zeros_like(tokens, dtype=np.bool_)
        
        for i in range(1, len(tokens)):
            if tokens[i] == PAD_TOKEN:
                break
            if random.random() < self.mask_prob:
                mask[i] = True
                rand = random.random()
                if rand < 0.8:
                    masked_tokens[i] = MASK_TOKEN
                elif rand < 0.9:
                    masked_tokens[i] = random.randint(SPECIAL_TOKENS, TOTAL_VOCAB_SIZE - 1)
        
        return masked_tokens, tokens, mask


class FlowClassificationDataset:
    def __init__(self, flow_data: List[Tuple[List[bytes], int]],
                 max_packets_per_flow: int = 3,
                 max_seq_len: int = 128):
        self.flow_data = flow_data
        self.max_packets_per_flow = max_packets_per_flow
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.flow_data)
    
    def __getitem__(self, idx):
        payloads, label = self.flow_data[idx]
        packet_tokens = []
        
        for i, payload in enumerate(payloads):
            if i >= self.max_packets_per_flow:
                break
            bigrams = bytes_to_bigrams(payload)
            
            if len(bigrams) > self.max_seq_len - 1:
                bigrams = bigrams[:self.max_seq_len - 1]
            
            tokens = [CLS_TOKEN] + bigrams
            if len(tokens) < self.max_seq_len:
                tokens += [PAD_TOKEN] * (self.max_seq_len - len(tokens))
            
            packet_tokens.append(np.array(tokens, dtype=np.int64))
        
        while len(packet_tokens) < self.max_packets_per_flow:
            packet_tokens.append(np.zeros(self.max_seq_len, dtype=np.int64))
        
        return np.array(packet_tokens), np.array(label, dtype=np.int64)