# Архитектуры Transformer моделей

## Обзор

Реализованы две архитектуры Transformer для классификации временных рядов:

1. **Encoder-Only Transformer** - простая и быстрая модель
2. **Time Series Transformer** - продвинутая модель с временным кодированием

## Encoder-Only Transformer

### Характеристики:
- **Скорость обучения**: Быстрая
- **Параметры**: ~2-4M (зависит от конфигурации)
- **Сложность**: Низкая
- **Применение**: Быстрое прототипирование, проверка гипотез

### Архитектура:
```
Input [batch, seq_len, features]
  ↓
Input Embedding [batch, seq_len, d_model]
  ↓
Positional Encoding
  ↓
Encoder Layers (4 слоя)
  ├─ Multi-Head Attention
  ├─ Feed-Forward
  └─ Residual Connections + LayerNorm
  ↓
Global Average Pooling
  ↓
Classifier [batch, num_classes]
```

### Параметры по умолчанию:
- `d_model`: 256
- `n_layers`: 4
- `n_heads`: 8
- `d_ff`: 1024
- `dropout`: 0.1

## Time Series Transformer

### Характеристики:
- **Скорость обучения**: Средняя
- **Параметры**: ~3-6M (зависит от конфигурации)
- **Сложность**: Высокая
- **Применение**: Финальная модель, максимальная точность

### Архитектура:
```
Input [batch, seq_len, features]
  ↓
Patch Embedding (опционально)
  ↓
Input Embedding [batch, seq_len, d_model]
  ↓
Temporal Encoding (временное кодирование)
  ↓
Encoder Layers (6 слоев)
  ├─ Multi-Head Attention
  ├─ Feed-Forward
  └─ Residual Connections + LayerNorm
  ↓
Multi-Scale Attention (опционально)
  ↓
Global Pooling (Mean + Max)
  ↓
Classifier [batch, num_classes]
```

### Особенности:
- **Temporal Encoding**: Учет временных меток (час, день недели)
- **Patch Embedding**: Группировка временных шагов (опционально)
- **Multi-Scale Attention**: Анализ на разных временных масштабах
- **Combined Pooling**: Mean + Max pooling для лучшего представления

### Параметры по умолчанию:
- `d_model`: 256
- `n_layers`: 6
- `n_heads`: 8
- `d_ff`: 1024
- `dropout`: 0.1
- `use_temporal_encoding`: True
- `use_patch_embedding`: False

## Использование

### Создание модели через фабрику:

```python
from models.model_factory import create_model, get_model_config
from config.model_config import TransformerConfig

# Вариант 1: Encoder-Only
config = get_model_config('encoder', num_features=100, num_classes=3)
model = create_model(config)

# Вариант 2: Time Series Transformer
config = get_model_config('timeseries', num_features=100, num_classes=3)
model = create_model(config)

# Вариант 3: Кастомная конфигурация
config = TransformerConfig(
    model_type='encoder',
    num_features=100,
    num_classes=3,
    d_model=512,
    n_layers=8,
    n_heads=16
)
model = create_model(config)
```

## Сравнение моделей

| Характеристика | Encoder-Only | Time Series |
|---------------|--------------|-------------|
| Скорость обучения | ⚡⚡⚡ Быстро | ⚡⚡ Средне |
| Точность | ⭐⭐⭐ Хорошо | ⭐⭐⭐⭐ Отлично |
| Параметры | ~2-4M | ~3-6M |
| Временное кодирование | ❌ | ✅ |
| Patch embedding | ❌ | ✅ (опционально) |
| Multi-scale | ❌ | ✅ |

## Рекомендации

1. **Начните с Encoder-Only** для быстрого прототипирования
2. **Используйте Time Series Transformer** для финальной модели
3. **Обучите обе модели** и сравните результаты
4. **Настройте гиперпараметры** под ваши данные

## Конфигурация

Все параметры настраиваются в `config/model_config.py`:

- `model_type`: 'encoder' или 'timeseries'
- `sequence_length`: длина последовательности (по умолчанию 60)
- `d_model`: размерность модели
- `n_layers`: количество слоев
- `n_heads`: количество attention heads
- `dropout`: уровень dropout
- `training_data_months`: количество месяцев данных (по умолчанию 6)

