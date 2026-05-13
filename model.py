import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from data_processing import TOTAL_VOCAB_SIZE, PAD_TOKEN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)
        
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        normed = self.norm1(x)
        attn_output, _ = self.self_attn(normed, normed, normed, mask)
        x = x + self.dropout1(attn_output)
        
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout2(ffn_output)
        
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, 
                 n_heads: int, d_ff: int, max_len: int = 512, dropout: float = 0.1,
                 pad_token_id: int = PAD_TOKEN, embedding_size: int = 128):
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.embedding_size = embedding_size
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_token_id)
        self.embedding_projection = nn.Linear(embedding_size, d_model)
        
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        self.shared_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.n_layers = n_layers
        
        self.norm = nn.LayerNorm(d_model)
    
    def get_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        return (x != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding_mask = self.get_padding_mask(x)
        
        x = self.token_embedding(x)
        x = self.embedding_projection(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for _ in range(self.n_layers):
            x = self.shared_layer(x, padding_mask)
        
        x = self.norm(x)
        return x


class PERTForMaskedLM(nn.Module):
    def __init__(self, vocab_size: int = TOTAL_VOCAB_SIZE, d_model: int = 256, 
                 n_layers: int = 6, n_heads: int = 8, d_ff: int = 1024, 
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input_ids)
        logits = self.mlm_head(encoder_output)
        return logits
    
    @torch.no_grad()
    def get_packet_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        encoder_output = self.encoder(input_ids)
        return encoder_output[:, 0, :]


class PERTForFlowClassification(nn.Module):
    def __init__(self, num_classes: int, vocab_size: int = TOTAL_VOCAB_SIZE, 
                 d_model: int = 256, n_layers: int = 6, n_heads: int = 8, 
                 d_ff: int = 1024, max_len: int = 128, max_packets: int = 3, 
                 dropout: float = 0.1, encoder: Optional[TransformerEncoder] = None):
        super().__init__()
        
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = TransformerEncoder(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                max_len=max_len,
                dropout=dropout
            )
        
        self.d_model = d_model
        self.max_packets = max_packets
        
        classifier_input_dim = d_model * max_packets
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, num_classes)
        )
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        
        packet_embeddings = []
        for i in range(self.max_packets):
            packet_ids = input_ids[:, i, :]
            with torch.set_grad_enabled(self.training):
                enc_out = self.encoder(packet_ids)
                cls_emb = enc_out[:, 0, :]
            packet_embeddings.append(cls_emb)
        
        concat_emb = torch.cat(packet_embeddings, dim=-1)
        logits = self.classifier(concat_emb)
        
        return logits
    
    @classmethod
    def from_pretrained(cls, checkpoint_path: str, num_classes: int, 
                        max_packets: int = 3, **kwargs):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model_state_dict = checkpoint.get('model_state_dict', checkpoint)
        encoder_state_dict = {}
        for k, v in model_state_dict.items():
            if k.startswith('encoder.'):
                encoder_state_dict[k[8:]] = v
        
        vocab_size = kwargs.pop('vocab_size', TOTAL_VOCAB_SIZE)
        d_model = kwargs.pop('d_model', 256)
        n_layers = kwargs.pop('n_layers', 6)
        n_heads = kwargs.pop('n_heads', 8)
        d_ff = kwargs.pop('d_ff', 1024)
        max_len = kwargs.pop('max_len', 128)
        dropout = kwargs.pop('dropout', 0.1)
        
        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )
        encoder.load_state_dict(encoder_state_dict)
        
        model = cls(
            num_classes=num_classes,
            encoder=encoder,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_len=max_len,
            max_packets=max_packets,
            dropout=dropout
        )
        
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Testing PERT model...")
    
    d_model = 256
    n_layers = 6
    n_heads = 8
    d_ff = 1024
    
    mlm_model = PERTForMaskedLM(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff
    )
    print(f"MLM Model parameters: {count_parameters(mlm_model):,}")
    
    batch_size = 4
    seq_len = 128
    dummy_input = torch.randint(0, TOTAL_VOCAB_SIZE, (batch_size, seq_len))
    logits = mlm_model(dummy_input)
    print(f"MLM output shape: {logits.shape}")
    
    num_classes = 12
    cls_model = PERTForFlowClassification(
        num_classes=num_classes,
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_packets=3
    )
    print(f"Classification Model parameters: {count_parameters(cls_model):,}")
    
    max_packets = 3
    dummy_flow_input = torch.randint(0, TOTAL_VOCAB_SIZE, (batch_size, max_packets, seq_len))
    cls_logits = cls_model(dummy_flow_input)
    print(f"Classification output shape: {cls_logits.shape}")
    
    print("Model test completed!")