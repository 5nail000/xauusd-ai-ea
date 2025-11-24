"""
Конфигурация для моделей Transformer
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TransformerConfig:
    """Конфигурация для Transformer модели"""
    
    # Тип модели
    model_type: str = 'encoder'  # 'encoder' или 'timeseries'
    
    # Параметры последовательности
    sequence_length: int = 60  # Длина последовательности (минуты)
    num_features: int = None  # Количество фичей (будет определено автоматически)
    
    # Параметры эмбеддингов
    d_model: int = 256  # Размерность модели
    d_ff: int = 1024  # Размерность feed-forward сети
    
    # Параметры attention
    n_heads: int = 8  # Количество attention heads
    dropout: float = 0.1  # Dropout rate
    
    # Параметры encoder
    n_layers: int = 6  # Количество encoder слоев
    
    # Параметры классификации
    num_classes: int = 3  # Количество классов (пробой, отскок, неопределенность)
    
    # Параметры для Time Series Transformer
    use_temporal_encoding: bool = True  # Использовать временное кодирование
    use_patch_embedding: bool = False  # Использовать patch embedding
    patch_size: int = 5  # Размер патча (если используется)
    
    # Параметры обучения
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Параметры данных
    training_data_months: int = 6  # Количество месяцев данных для обучения (по умолчанию)
    
    def __post_init__(self):
        """Валидация параметров"""
        if self.model_type not in ['encoder', 'timeseries']:
            raise ValueError(f"Неизвестный тип модели: {self.model_type}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) должно быть кратно n_heads ({self.n_heads})")

# Конфигурации по умолчанию для разных типов моделей
encoder_config = TransformerConfig(
    model_type='encoder',
    d_model=256,
    n_layers=4,
    n_heads=8,
    dropout=0.1
)

timeseries_config = TransformerConfig(
    model_type='timeseries',
    d_model=256,
    n_layers=6,
    n_heads=8,
    dropout=0.1,
    use_temporal_encoding=True,
    use_patch_embedding=False
)

