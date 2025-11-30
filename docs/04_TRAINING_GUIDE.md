# Руководство по обучению модели

## Обзор

Полный процесс обучения Transformer модели для классификации торговых сигналов на данных по золоту.

## Шаги обучения

### Вариант 1: Полный цикл (рекомендуется)

Используйте единый скрипт для всего процесса:

```bash
# Полный цикл: подготовка данных -> обучение -> бэктестинг
python full_pipeline.py --months 12

# С параметрами
python full_pipeline.py --months 12 --encoder-only --epochs 50

# С удалением высококоррелированных фичей
# Примечание: анализ выполняется на объединенном датасете (train+val+test)
# для гарантии одинакового набора фичей во всех файлах
python full_pipeline.py --months 12 --remove-correlated

# С кастомным порогом корреляции
python full_pipeline.py --months 12 --remove-correlated --correlation-threshold 0.90

# Пропустить этапы (если данные/модели уже есть)
python full_pipeline.py --skip-prepare  # Пропустить подготовку данных
python full_pipeline.py --skip-train     # Пропустить обучение
python full_pipeline.py --skip-backtest # Пропустить бэктестинг
```

### Вариант 2: По отдельности

#### 1. Подготовка данных

Сначала подготовьте данные:

```bash
python prepare_gold_data.py
```

Это создаст:
- `workspace/prepared/features/gold_train.csv` - обучающая выборка
- `workspace/prepared/features/gold_val.csv` - валидационная выборка
- `workspace/prepared/features/gold_test.csv` - тестовая выборка

#### 2. Обучение моделей

#### Обучение обеих моделей (рекомендуется):

```bash
python train_all_models.py
```

Этот скрипт автоматически:
1. Загрузит данные
2. Обучит encoder модель
3. Обучит timeseries модель
4. Сохранит все результаты

#### Обучение одной модели:

```bash
python train_model.py
```

Или с параметрами:

```bash
# Только encoder модель
python train_all_models.py --encoder-only

# Только timeseries модель
python train_all_models.py --timeseries-only

# С параметрами
python train_all_models.py --months 12 --epochs 50 --batch-size 16
```

#### Параметры командной строки:

- `-m, --months` - количество месяцев данных (по умолчанию: 12)
- `--encoder-only` - обучить только encoder модель
- `--timeseries-only` - обучить только timeseries модель
- `--batch-size` - размер батча (по умолчанию: 32)
- `--epochs` - количество эпох (по умолчанию: 100)
- `--patience` - терпение для early stopping (по умолчанию: 10)

### 3. Настройка параметров

Можно изменить параметры через командную строку или в `train_model.py`:

**Через командную строку:**
```bash
python train_model.py \
  --dropout 0.2 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --patience 10 \
  --epochs 100 \
  --batch-size 32
```

**Или изменить параметры в `train_model.py`:**

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

После обучения автоматически создаются:

1. **Модель**: `workspace/models/checkpoints/{model_type}_model.pth`
   - Содержит веса модели и конфигурацию
2. **Scaler**: `workspace/prepared/scalers/feature_scaler_{model_type}.pkl`
   - Отдельный scaler для каждого типа модели
   - **Включает метаданные**: список фичей, настройки подготовки, hash фичей
3. **Документация по фичам**: ⭐ НОВОЕ
   - `workspace/models/checkpoints/{model_type}_model_features_documentation.json` - структурированные данные
   - `workspace/models/checkpoints/{model_type}_model_features_documentation.md` - удобная для чтения документация
   - Содержит описание, формулу, источник и статистику для каждого фича
4. **История обучения**:
   - `workspace/models/metrics/{model_type}_model_history.pkl` - pickle файл
   - `workspace/models/metrics/{model_type}_model_history.csv` - CSV с метриками по эпохам
5. **Графики обучения**: `workspace/models/metrics/{model_type}_model_training_curves.png`
   - Loss, Accuracy, Learning Rate по эпохам
6. **Confusion Matrix**: `workspace/models/metrics/confusion_matrix_{model_type}.png`
7. **Анализ переобученности**: выводится в терминал после обучения

## Использование обученной модели

```python
import torch
from models.model_factory import create_model, get_model_config
from models.data_loader import SequenceGenerator

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = get_model_config('encoder', num_features=100, num_classes=3)
model = create_model(config)
model.load_state_dict(torch.load('workspace/models/checkpoints/encoder_model.pth', map_location=device))
model.eval()

# Загрузка scaler
seq_gen = SequenceGenerator(sequence_length=60)
seq_gen.load_scaler('workspace/prepared/scalers/feature_scaler.pkl')

# Предсказание
# (подготовьте данные и создайте последовательности)
with torch.no_grad():
    predictions = model(sequences)
    _, predicted_classes = torch.max(predictions, 1)
```

## Сравнение моделей

### Автоматическое обучение обеих моделей

```bash
python train_all_models.py
```

Скрипт автоматически:
1. Обучит encoder модель
2. Обучит timeseries модель
3. Выведет итоговую сводку с метриками обеих моделей

### Ручное сравнение

После обучения обеих моделей сравните:

1. **Метрики в терминале** - итоговая сводка после `train_all_models.py`
2. **CSV файлы истории** - сравните кривые обучения
3. **Графики** - визуальное сравнение кривых
4. **Confusion Matrix** - сравните ошибки классификации

### Выбор лучшей модели

Рекомендуется:
- **Encoder** - если важна скорость и простота
- **TimeSeries** - если важна максимальная точность
- Сравните метрики на тестовой выборке для финального решения

## Мониторинг обучения

### Во время обучения

В терминале выводятся:
- **Progress bar** для каждого батча (loss, accuracy)
- **Метрики после каждой эпохи**:
  - Train Loss и Accuracy
  - Val Loss и Accuracy
  - Learning Rate
- **Лучшая валидационная точность**
- **Early stopping** (если сработал)

### После обучения

Автоматически создаются:

1. **CSV файл с историей** (`*_history.csv`):
   - Все метрики по эпохам
   - Gap между train и val (для анализа переобученности)
   - Можно анализировать в Excel/Pandas

2. **Графики обучения** (`*_training_curves.png`):
   - Loss (train vs val)
   - Accuracy (train vs val)
   - Learning Rate

3. **Анализ переобученности**:
   - Автоматическое определение переобученности
   - Оценка степени (Нет/Слабое/Умеренное/Сильное)
   - Рекомендации по улучшению

### Анализ истории обучения

```python
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка истории
history = pd.read_csv('workspace/models/metrics/encoder_model_history.csv')

# Построение графиков
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

# Анализ переобученности
gap = history['val_acc'] - history['train_acc']
print(f"Средний gap: {gap.mean():.2f}%")
print(f"Максимальный gap: {gap.max():.2f}%")
```

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

