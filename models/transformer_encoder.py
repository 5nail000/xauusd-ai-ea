"""
Encoder-Only Transformer для классификации временных рядов
Простая и быстрая архитектура
"""
import torch
import torch.nn as nn
from typing import Optional
from models.base_transformer import BaseTransformer

class EncoderTransformer(BaseTransformer):
    """
    Encoder-Only Transformer для классификации
    
    Простая архитектура, быстро обучается, хорошо работает для классификации
    временных рядов.
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # Дополнительные слои для encoder-only модели
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_features]
            mask: Optional attention mask
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Input embedding
        x = self.input_embedding(x)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Global average pooling (можно также использовать max pooling или последний элемент)
        # Вариант 1: Mean pooling
        x = x.mean(dim=1)
        
        # Вариант 2: Max pooling (раскомментировать если нужно)
        # x = x.max(dim=1)[0]
        
        # Вариант 3: Last element (раскомментировать если нужно)
        # x = x[:, -1, :]
        
        # Classification
        logits = self.classifier(x)
        
        return logits

