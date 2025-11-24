# Руководство по обучению модели

## Обзор

Полный процесс обучения Transformer модели для классификации торговых сигналов на данных по золоту.

## Шаги обучения

### 1. Подготовка данных

Сначала подготовьте данные:

```bash
python prepare_gold_data.py
```

Это создаст:
- `data/gold_train.csv` - обучающая выборка
- `data/gold_val.csv` - валидационная выборка
- `data/gold_test.csv` - тестовая выборка

### 2. Обучение модели

Запустите обучение:

```bash
python train_model.py
```

Скрипт автоматически:
1. Загрузит данные
2. Создаст последовательности для Transformer
3. Создаст модель (Encoder-Only по умолчанию)
4. Обучит модель с early stopping
5. Оценит модель на всех выборках
6. Сохранит результаты

### 3. Настройка параметров

Можно изменить параметры в `train_model.py`:

```python
# Тип модели
model_type = 'encoder'  # или 'timeseries'

# Параметры модели
config = get_model_config(
    model_type=model_type,
    num_features=...,  # Автоматически определяется
    num_classes=3,
    sequence_length=60,
    d_model=256,
    n_layers=4,
    n_heads=8,
    dropout=0.1,
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=100,
    early_stopping_patience=10
)
```

## Структура обучения

### DataLoader

Модуль `models/data_loader.py` создает:
- Последовательности длиной 60 минут
- Нормализованные фичи
- DataLoader'ы для train/val/test

### Trainer

Модуль `models/trainer.py` обеспечивает:
- Обучение с оптимизатором AdamW
- Learning rate scheduling
- Early stopping
- Сохранение лучших весов
- Мониторинг метрик

### Callbacks

Модуль `models/callbacks.py` содержит:
- **EarlyStopping**: остановка при отсутствии улучшений
- **ModelCheckpoint**: сохранение лучших весов
- **LearningRateScheduler**: изменение learning rate
- **TrainingHistory**: история обучения

### Evaluator

Модуль `models/evaluator.py` предоставляет:
- Метрики (accuracy, precision, recall, F1)
- Confusion matrix
- Classification report
- Визуализацию результатов

## Результаты обучения

После обучения создаются:

1. **Модель**: `models/checkpoints/{model_type}_model.pth`
2. **Scaler**: `models/feature_scaler.pkl`
3. **История**: `models/checkpoints/{model_type}_model_history.pkl`
4. **Confusion Matrix**: `models/confusion_matrix_{model_type}.png`

## Использование обученной модели

```python
import torch
from models.model_factory import create_model, get_model_config
from models.data_loader import SequenceGenerator

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = get_model_config('encoder', num_features=100, num_classes=3)
model = create_model(config)
model.load_state_dict(torch.load('models/checkpoints/encoder_model.pth', map_location=device))
model.eval()

# Загрузка scaler
seq_gen = SequenceGenerator(sequence_length=60)
seq_gen.load_scaler('models/feature_scaler.pkl')

# Предсказание
# (подготовьте данные и создайте последовательности)
with torch.no_grad():
    predictions = model(sequences)
    _, predicted_classes = torch.max(predictions, 1)
```

## Сравнение моделей

Можно обучить обе модели и сравнить:

```python
# Encoder-Only (быстрая)
model_type = 'encoder'
# ... обучение ...

# Time Series Transformer (продвинутая)
model_type = 'timeseries'
# ... обучение ...
```

Затем сравните метрики на тестовой выборке.

## Мониторинг обучения

Во время обучения выводятся:
- Loss и accuracy на каждой эпохе
- Learning rate
- Лучшая валидационная точность
- Early stopping (если сработал)

## Советы

1. **Начните с Encoder-Only** для быстрого прототипирования
2. **Используйте GPU** если доступно (автоматически определяется)
3. **Настройте batch_size** в зависимости от доступной памяти
4. **Мониторьте метрики** на валидационной выборке
5. **Сохраняйте историю** для анализа обучения

## Следующие шаги

После обучения:
1. Проанализируйте confusion matrix
2. Проверьте метрики по классам
3. Проведите бэктестинг стратегии
4. Оптимизируйте параметры входа/выхода

