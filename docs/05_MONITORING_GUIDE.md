# Руководство по мониторингу обучения

## Обзор

Система мониторинга обучения предоставляет детальную информацию о процессе обучения модели, включая метрики по эпохам, визуализацию кривых обучения и анализ переобученности.

## Автоматический мониторинг

### Во время обучения

В терминале выводятся:

1. **Progress bar (tqdm)** для каждого батча:
   - Текущий loss
   - Текущая accuracy
   - Скорость обработки (it/s)

2. **Метрики после каждой эпохи**:
   ```
   Эпоха 1/100
   ------------------------------------------------------------
   Train Loss: 0.0199, Train Acc: 99.72%
   Val Loss:   0.0001, Val Acc:   100.00%
   Learning Rate: 0.000100
   ```

3. **Early stopping** (если сработал):
   ```
   Early stopping на эпохе 4
   Лучшая валидационная точность: 100.00%
   ```

### После обучения

Автоматически создаются:

1. **CSV файл с историей** (`*_history.csv`):
   - Все метрики по эпохам
   - Колонки: `epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`
   - Дополнительные метрики: `loss_gap`, `acc_gap`, `overfitting_score`

2. **Графики обучения** (`*_training_curves.png`):
   - Loss (train vs val)
   - Accuracy (train vs val)
   - Learning Rate

3. **Анализ переобученности**:
   - Автоматическое определение переобученности
   - Оценка степени (Нет/Слабое/Умеренное/Сильное)
   - Рекомендации по улучшению

## Анализ истории обучения

### Загрузка истории

```python
import pandas as pd

# Загрузка из CSV
history = pd.read_csv('models/checkpoints/encoder_model_history.csv')

# Или из pickle
from models.callbacks import TrainingHistory
history_obj = TrainingHistory()
history_obj.load('models/checkpoints/encoder_model_history.pkl')
history_dict = history_obj.history
```

### Построение графиков

```python
import matplotlib.pyplot as plt

history = pd.read_csv('models/checkpoints/encoder_model_history.csv')

# Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
plt.plot(history['epoch'], history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Model Accuracy')

plt.tight_layout()
plt.show()
```

### Анализ переобученности

```python
history = pd.read_csv('models/checkpoints/encoder_model_history.csv')

# Вычисляем gap
history['acc_gap'] = history['val_acc'] - history['train_acc']
history['loss_gap'] = history['train_loss'] - history['val_loss']

# Анализ
final_gap = history['acc_gap'].iloc[-1]
avg_gap = history['acc_gap'].mean()
max_gap = history['acc_gap'].max()

print(f"Финальный gap: {final_gap:.2f}%")
print(f"Средний gap: {avg_gap:.2f}%")
print(f"Максимальный gap: {max_gap:.2f}%")

# Определение переобученности
if final_gap < -5:
    print("⚠️ Обнаружена переобученность!")
    if final_gap < -20:
        print("  Степень: Сильное")
    elif final_gap < -10:
        print("  Степень: Умеренное")
    else:
        print("  Степень: Слабое")
else:
    print("✓ Переобученность не обнаружена")
```

### Использование встроенного анализа

```python
from models.callbacks import TrainingHistory

history = TrainingHistory()
history.load('models/checkpoints/encoder_model_history.pkl')

# Автоматический анализ
analysis = history.analyze_overfitting()

print(f"Переобученность: {analysis['overfitting_severity']}")
print(f"Финальный gap: {analysis['final_acc_gap']:.2f}%")
print(f"Лучшая валидационная точность: {analysis['best_val_acc']:.2f}%")
print(f"Достигнута на эпохе: {analysis['best_val_acc_epoch']}")

# Построение графиков
history.plot_history(save_path='my_training_curves.png', show=True)
```

## Метрики для анализа

### Основные метрики

- **Train Loss/Accuracy**: метрики на обучающей выборке
- **Val Loss/Accuracy**: метрики на валидационной выборке
- **Gap**: разница между train и val метриками
- **Learning Rate**: изменение learning rate по эпохам

### Метрики переобученности

- **Loss Gap**: `train_loss - val_loss`
  - Отрицательное = train loss выше val loss (хорошо)
  - Положительное = train loss ниже val loss (возможна переобученность)

- **Accuracy Gap**: `val_acc - train_acc`
  - Положительное = val accuracy выше train (хорошо)
  - Отрицательное = train accuracy выше val (переобученность)

- **Overfitting Score**: равен `acc_gap`
  - < -20: Сильное переобучение
  - -20 до -10: Умеренное переобучение
  - -10 до -5: Слабое переобучение
  - > -5: Переобучение не обнаружено

## Интерпретация результатов

### Нормальное обучение

- Train и Val метрики близки друг к другу
- Обе метрики улучшаются со временем
- Gap остается небольшим (< 5%)

### Переобучение

- Train метрики значительно лучше Val метрики
- Val метрики перестают улучшаться или ухудшаются
- Gap увеличивается со временем

**Рекомендации при переобучении:**
- Увеличить dropout
- Добавить регуляризацию (weight decay)
- Увеличить размер обучающей выборки
- Уменьшить сложность модели
- Использовать data augmentation

### Недообучение

- Обе метрики низкие и не улучшаются
- Gap небольшой, но метрики плохие

**Рекомендации при недообучении:**
- Увеличить сложность модели
- Уменьшить dropout
- Увеличить количество эпох
- Улучшить фичи

## Сравнение моделей

### Сравнение кривых обучения

```python
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка истории обеих моделей
encoder_history = pd.read_csv('models/checkpoints/encoder_model_history.csv')
timeseries_history = pd.read_csv('models/checkpoints/timeseries_model_history.csv')

# Сравнение accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(encoder_history['epoch'], encoder_history['val_acc'], label='Encoder Val')
plt.plot(timeseries_history['epoch'], timeseries_history['val_acc'], label='TimeSeries Val')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.legend()
plt.title('Validation Accuracy Comparison')

# Сравнение loss
plt.subplot(1, 2, 2)
plt.plot(encoder_history['epoch'], encoder_history['val_loss'], label='Encoder Val')
plt.plot(timeseries_history['epoch'], timeseries_history['val_loss'], label='TimeSeries Val')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.title('Validation Loss Comparison')

plt.tight_layout()
plt.show()
```

## Файлы мониторинга

После обучения создаются:

- `models/checkpoints/{model_type}_model_history.csv` - CSV с метриками
- `models/checkpoints/{model_type}_model_history.pkl` - Pickle файл (для загрузки в Python)
- `models/checkpoints/{model_type}_model_training_curves.png` - Графики обучения
- `models/confusion_matrix_{model_type}.png` - Confusion matrix

## Советы

1. **Регулярно проверяйте графики** - они показывают проблемы раньше, чем метрики
2. **Следите за gap** - увеличение gap = признак переобучения
3. **Сравнивайте модели** - используйте графики для выбора лучшей модели
4. **Сохраняйте историю** - она нужна для анализа и сравнения экспериментов
5. **Используйте early stopping** - он предотвращает переобучение

