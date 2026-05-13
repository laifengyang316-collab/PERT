import torch
import numpy as np
from typing import List, Tuple

from model import PERTForFlowClassification
from data_processing import (
    bytes_to_bigrams,
    TOTAL_VOCAB_SIZE,
    CLS_TOKEN,
    PAD_TOKEN
)


class PERTClassifier:
    def __init__(self, checkpoint_path: str, num_classes: int, max_packets: int = 3,
                 max_seq_len: int = 128, d_model: int = 256, n_layers: int = 6,
                 n_heads: int = 8, d_ff: int = 1024, dropout: float = 0.1,
                 device: str = None):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.max_packets = max_packets
        self.max_seq_len = max_seq_len
        
        print(f"Loading model from {checkpoint_path}...")
        self.model = PERTForFlowClassification.from_pretrained(
            checkpoint_path,
            num_classes=num_classes,
            max_packets=max_packets,
            vocab_size=TOTAL_VOCAB_SIZE,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def preprocess_flow(self, payloads: List[bytes]) -> torch.Tensor:
        packet_tokens = []
        
        for i, payload in enumerate(payloads):
            if i >= self.max_packets:
                break
            
            bigrams = bytes_to_bigrams(payload)
            if len(bigrams) > self.max_seq_len - 1:
                bigrams = bigrams[:self.max_seq_len - 1]
            
            tokens = [CLS_TOKEN] + bigrams
            if len(tokens) < self.max_seq_len:
                tokens += [PAD_TOKEN] * (self.max_seq_len - len(tokens))
            
            packet_tokens.append(np.array(tokens, dtype=np.int64))
        
        while len(packet_tokens) < self.max_packets:
            packet_tokens.append(np.zeros(self.max_seq_len, dtype=np.int64))
        
        input_tensor = torch.tensor(np.array(packet_tokens), dtype=torch.long)
        return input_tensor.unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, payloads: List[bytes]) -> Tuple[int, np.ndarray]:
        input_tensor = self.preprocess_flow(payloads).to(self.device)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        
        pred_class = torch.argmax(probs, dim=-1).item()
        pred_probs = probs.cpu().numpy()[0]
        
        return pred_class, pred_probs
    
    @torch.no_grad()
    def predict_batch(self, flows: List[List[bytes]]) -> Tuple[List[int], np.ndarray]:
        batch_tensors = []
        
        for payloads in flows:
            input_tensor = self.preprocess_flow(payloads)
            batch_tensors.append(input_tensor)
        
        batch_input = torch.cat(batch_tensors, dim=0).to(self.device)
        logits = self.model(batch_input)
        probs = torch.softmax(logits, dim=-1)
        
        pred_classes = torch.argmax(probs, dim=-1).cpu().numpy().tolist()
        pred_probs = probs.cpu().numpy()
        
        return pred_classes, pred_probs


if __name__ == "__main__":
    print("PERT Classifier Inference Tool")
    print("Usage: from inference import PERTClassifier")
    print()
    print("Example:")
    print("  classifier = PERTClassifier(")
    print("      checkpoint_path='./classifier_checkpoints/classifier_best.pt',")
    print("      num_classes=12,")
    print("      max_packets=3,")
    print("      max_seq_len=128")
    print("  )")
    print("  pred_class, probs = classifier.predict(payloads)")