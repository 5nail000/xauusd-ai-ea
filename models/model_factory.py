"""
Фабрика для создания Transformer моделей
"""
from models.transformer_encoder import EncoderTransformer
from models.transformer_timeseries import TimeSeriesTransformer
from config.model_config import TransformerConfig

def create_model(config: TransformerConfig):
    """
    Создает модель Transformer на основе конфигурации
    
    Args:
        config: Конфигурация модели
    
    Returns:
        Модель Transformer
    """
    if config.model_type == 'encoder':
        model = EncoderTransformer(config)
    elif config.model_type == 'timeseries':
        model = TimeSeriesTransformer(config)
    else:
        raise ValueError(f"Неизвестный тип модели: {config.model_type}")
    
    return model

def get_model_config(model_type: str = 'encoder', **kwargs) -> TransformerConfig:
    """
    Получает конфигурацию для модели
    
    Args:
        model_type: Тип модели ('encoder' или 'timeseries')
        **kwargs: Дополнительные параметры для переопределения
    
    Returns:
        Конфигурация модели
    """
    if model_type == 'encoder':
        config = TransformerConfig(
            model_type='encoder',
            d_model=256,
            n_layers=4,
            n_heads=8,
            dropout=0.1
        )
    elif model_type == 'timeseries':
        config = TransformerConfig(
            model_type='timeseries',
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            use_temporal_encoding=True,
            use_patch_embedding=False
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    # Переопределяем параметры из kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config

