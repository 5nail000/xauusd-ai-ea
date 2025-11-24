"""
Time Series Transformer для классификации временных рядов
Продвинутая архитектура с временным кодированием
"""
import torch
import torch.nn as nn
import math
from typing import Optional
from models.base_transformer import BaseTransformer, TransformerEncoderLayer

class TemporalEncoding(nn.Module):
    """Временное кодирование для учета временных меток"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Создаем позиционное кодирование для времени
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            time_features: [batch_size, seq_len, time_dim] - временные фичи (час, день недели и т.д.)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        
        # Если есть временные фичи, добавляем их
        if time_features is not None:
            time_proj = nn.Linear(time_features.size(-1), self.d_model).to(x.device)
            time_emb = time_proj(time_features)
            x = x + time_emb
        
        return x

class PatchEmbedding(nn.Module):
    """Patch embedding для группировки временных шагов"""
    
    def __init__(self, patch_size: int, d_model: int, num_features: int):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size * num_features, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, num_features]
        
        Returns:
            [batch_size, new_seq_len, d_model]
        """
        batch_size, seq_len, num_features = x.size()
        
        # Разбиваем на патчи
        num_patches = seq_len // self.patch_size
        x = x[:, :num_patches * self.patch_size, :]
        x = x.view(batch_size, num_patches, self.patch_size * num_features)
        
        # Проекция
        x = self.projection(x)
        return x

class TimeSeriesTransformer(BaseTransformer):
    """
    Time Series Transformer для классификации
    
    Продвинутая архитектура с:
    - Временным кодированием
    - Опциональным patch embedding
    - Multi-scale attention
    """
    
    def __init__(self, config):
        # Сохраняем оригинальную размерность
        self.original_num_features = config.num_features
        
        # Если используется patch embedding, изменяем входную размерность
        if config.use_patch_embedding:
            self.patch_embedding = PatchEmbedding(
                config.patch_size,
                config.d_model,
                config.num_features
            )
            # После patch embedding размерность фичей меняется
            # Но для базового класса оставляем оригинальную
            self.use_patch = True
        else:
            self.patch_embedding = None
            self.use_patch = False
        
        super().__init__(config)
        
        # Временное кодирование
        if config.use_temporal_encoding:
            self.temporal_encoding = TemporalEncoding(
                config.d_model,
                max_len=config.sequence_length
            )
        else:
            self.temporal_encoding = None
        
        # Multi-scale attention (опционально)
        self.multi_scale_attention = None
        if config.n_layers > 4:
            # Добавляем дополнительный слой для multi-scale анализа
            self.multi_scale_attention = TransformerEncoderLayer(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.dropout
            )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                time_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, num_features]
            mask: Optional attention mask
            time_features: Optional temporal features [batch_size, seq_len, time_dim]
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # Patch embedding (если используется)
        if self.use_patch and self.patch_embedding is not None:
            x = self.patch_embedding(x)
        else:
            # Обычное input embedding
            x = self.input_embedding(x)
        
        # Временное кодирование
        if self.temporal_encoding is not None:
            x = self.temporal_encoding(x, time_features)
        else:
            # Обычное позиционное кодирование
            x = self.pos_encoding(x)
        
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Multi-scale attention (если используется)
        if self.multi_scale_attention is not None:
            x = self.multi_scale_attention(x, mask)
        
        # Global pooling - используем комбинацию mean и max
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1)[0]
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Увеличиваем размерность для классификатора
        if x.size(-1) != self.config.d_model:
            x = nn.Linear(x.size(-1), self.config.d_model).to(x.device)(x)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

