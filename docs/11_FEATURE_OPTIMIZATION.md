# Оптимизация фичей и синхронизация

## Обзор

Руководство по оптимизации фичей для улучшения качества модели: анализ корреляции, удаление дубликатов, мониторинг аномалий и синхронизация фичей между обучением и применением.

## Анализ корреляции фичей

### Проблема мультиколлинеарности

Высокая корреляция между фичами (>0.95) может привести к:
- Избыточности информации
- Ухудшению обобщения модели
- Увеличению времени обучения
- Проблемам с интерпретацией

### Автоматический анализ

Используйте скрипт `analyze_feature_correlation.py`:

```bash
# Базовый анализ (порог 0.95)
python analyze_feature_correlation.py

# С другим порогом
python analyze_feature_correlation.py --threshold 0.90

# С визуализацией
python analyze_feature_correlation.py --plot

# Автоматическое удаление высококоррелированных фичей
python analyze_feature_correlation.py --remove
```

### Параметры скрипта

- `--input` - путь к файлу с данными (по умолчанию: `workspace/prepared/features/gold_train.csv`)
- `--threshold` - порог корреляции для удаления (по умолчанию: 0.95)
- `--remove` - автоматически удалить высококоррелированные фичи
- `--plot` - построить график корреляционной матрицы
- `--output` - путь для сохранения очищенных данных

### Стратегия удаления

Скрипт автоматически выбирает фичи для удаления на основе приоритетов:

1. **Высокий приоритет** (оставляем):
   - Базовые ценовые фичи: `close`, `open`, `high`, `low`, `returns`, `log_returns`

2. **Средний приоритет**:
   - Технические индикаторы: `sma`, `ema`, `rsi`, `macd`, `atr`

3. **Низкий приоритет** (удаляем первыми):
   - Производные фичи: `lag`, `stat`, `tick`, `multitimeframe`, `position`, `shadow`

### Автоматическое удаление в pipeline

Включите автоматическое удаление в конфигурации:

```python
from config.feature_config import FeatureConfig

config = FeatureConfig(
    remove_correlated_features=True,  # Включить удаление
    correlation_threshold=0.95        # Порог корреляции
)
```

При подготовке данных высококоррелированные фичи будут удалены автоматически.

## Мониторинг аномалий

### Проблема выхода за диапазон обучения

Если цена выходит за диапазон, на котором обучалась модель, входные значения могут стать аномальными (выходят за 3σ от обучающей выборки).

### Автоматический мониторинг

Система автоматически отслеживает аномалии во время бэктестинга:

1. **Проверка аномалий**: для каждого сигнала проверяется, выходят ли значения за 3σ
2. **Снижение уверенности**: если обнаружена аномалия, уверенность модели снижается пропорционально
3. **Пропуск сигналов**: если >50% фичей аномальны, сигнал пропускается

### Статистика аномалий

После бэктестинга выводится статистика:

```
⚠️  Мониторинг аномалий:
  Всего проверок: 1000
  Обнаружено аномалий: 45 (4.5%)
  Сигналов пропущено: 12
  Уверенность снижена: 33 раз
```

### Настройка порога

Порог аномалий можно изменить в `Backtester`:

```python
backtester = Backtester(
    model_path='workspace/models/checkpoints/encoder_model.pth',
    scaler_path='workspace/prepared/scalers/feature_scaler_encoder.pkl'
)
backtester.anomaly_threshold = 2.5  # 2.5σ вместо 3σ
```

## Рекомендации

### Для анализа корреляции

1. **Регулярно проверяйте корреляцию** после добавления новых фичей
2. **Используйте порог 0.95** для нейросетей (можно снизить до 0.90 для линейных моделей)
3. **Визуализируйте корреляции** для понимания структуры данных
4. **Автоматизируйте удаление** в pipeline для новых данных

### Для мониторинга аномалий

1. **Расширяйте обучающую выборку** - включайте периоды с разными ценовыми уровнями
2. **Используйте RobustScaler** вместо StandardScaler для устойчивости к выбросам
3. **Мониторьте статистику** - высокая доля аномалий (>10%) указывает на проблему
4. **Настройте порог** под ваши данные и требования

### Для улучшения обобщения

1. **Используйте относительные фичи** - процентные изменения вместо абсолютных
2. **Нормализуйте данные** - StandardScaler или RobustScaler
3. **Удаляйте дубликаты** - высококоррелированные фичи
4. **Мониторьте аномалии** - отслеживайте выход за диапазон обучения

## Примеры использования

### Анализ корреляции перед обучением

```bash
# 1. Подготовка данных
python prepare_gold_data.py --months 12

# 2. Анализ корреляции
python analyze_feature_correlation.py --plot

# 3. Удаление коррелированных фичей
python analyze_feature_correlation.py --remove --threshold 0.95

# 4. Обучение на очищенных данных
python train_all_models.py
```

### Автоматическое удаление в pipeline

```python
from config.feature_config import FeatureConfig
from data.gold_data_prep import GoldDataPreparator

# Конфигурация с автоматическим удалением
config = FeatureConfig(
    remove_correlated_features=True,
    correlation_threshold=0.95
)

preparator = GoldDataPreparator(config=config, training_months=12)
df = preparator.prepare_full_dataset(symbol='XAUUSD', months=12)
```

### Мониторинг аномалий в бэктестинге

```python
from trading.backtester import Backtester

backtester = Backtester(
    model_path='workspace/models/checkpoints/encoder_model.pth',
    scaler_path='workspace/prepared/scalers/feature_scaler_encoder.pkl'
)

# Бэктестинг с автоматическим мониторингом аномалий
results = backtester.backtest(test_df)

# Статистика аномалий доступна в:
print(backtester.anomaly_stats)
```

## Синхронизация фичей между обучением и применением

### Проблема

При разных настройках подготовки данных (например, с/без удаления коррелированных фичей) могут получиться разные наборы фичей. Это критично для корректной работы модели.

### Решение

Система автоматически сохраняет и проверяет соответствие фичей:

1. **При обучении** (`models/data_loader.py`):
   - Список фичей сохраняется в scaler файл
   - Сохраняются метаданные о настройках подготовки
   - Сохраняется hash фичей для проверки целостности

2. **При применении** (`trading/backtester.py`):
   - Автоматическая валидация фичей при загрузке scaler
   - Проверка наличия всех необходимых фичей
   - Предупреждения о лишних фичах
   - Использование только сохраненных фичей в правильном порядке

### Метаданные в scaler

Каждый scaler файл содержит:

```python
{
    'scaler': StandardScaler,
    'feature_columns': ['close', 'returns', 'rsi_14', ...],  # Список фичей
    'feature_stats': {...},  # Статистика для мониторинга аномалий
    'metadata': {
        'training_months': 12,
        'model_type': 'encoder',
        'num_features': 608,
        'preparation_config': {
            'remove_correlated_features': True,
            'correlation_threshold': 0.95
        },
        'feature_columns_hash': 1234567890,  # Hash для проверки
        'saved_at': '2025-11-24T23:30:00'
    }
}
```

### Валидация фичей

#### Автоматическая валидация

При бэктестинге валидация происходит автоматически:

```python
backtester = Backtester(
    model_path='workspace/models/checkpoints/encoder_model.pth',
    scaler_path='workspace/prepared/scalers/feature_scaler_encoder.pkl'
)
# Валидация происходит автоматически при вызове backtest()
results = backtester.backtest(test_df)
```

#### Ручная валидация

Используйте скрипт `validate_features.py`:

```bash
# Валидация test данных
python validate_features.py

# Валидация train данных
python validate_features.py --data workspace/prepared/features/gold_train.csv

# С другим scaler
python validate_features.py --scaler workspace/prepared/scalers/feature_scaler_timeseries.pkl
```

#### Программная валидация

```python
from utils.feature_validator import validate_dataframe_features, print_validation_report

result = validate_dataframe_features(
    df=test_df,
    scaler_path='workspace/prepared/scalers/feature_scaler_encoder.pkl'
)

print_validation_report(result)
```

### Что происходит при несоответствии

1. **Отсутствующие фичи** → `ValueError`:
   ```
   ❌ ОШИБКА: В DataFrame отсутствуют фичи, которые использовались при обучении:
      Отсутствуют: ['feature1', 'feature2', ...]
      Всего отсутствует: 15 из 608
   ```

2. **Лишние фичи** → Предупреждение (игнорируются):
   ```
   ⚠️  ПРЕДУПРЕЖДЕНИЕ: В DataFrame есть фичи, которых не было при обучении:
      Лишние фичи: ['extra_feature1', ...]
      Эти фичи будут проигнорированы
   ```

3. **Неправильный порядок** → Автоматически исправляется

### Рекомендации для разных экспериментов

#### Вариант 1: Версионирование файлов

Используйте разные имена для разных экспериментов:

```python
# Эксперимент 1: без удаления коррелированных
scaler_path = 'workspace/prepared/scalers/feature_scaler_encoder_v1.pkl'
model_path = 'workspace/models/checkpoints/encoder_model_v1.pth'

# Эксперимент 2: с удалением коррелированных
scaler_path = 'workspace/prepared/scalers/feature_scaler_encoder_v2_corr.pkl'
model_path = 'workspace/models/checkpoints/encoder_model_v2_corr.pth'
```

#### Вариант 2: Использование метаданных

Проверяйте метаданные перед использованием:

```python
from utils.feature_validator import load_scaler_metadata

metadata = load_scaler_metadata('workspace/prepared/scalers/feature_scaler_encoder.pkl')
if metadata.get('preparation_config', {}).get('remove_correlated_features'):
    print("Эта модель обучена с удалением коррелированных фичей")
```

#### Вариант 3: Документирование

Ведите журнал экспериментов:

```
experiments/
├── exp1_no_corr/
│   ├── model.pth
│   ├── scaler.pkl
│   └── config.txt
└── exp2_with_corr/
    ├── model.pth
    ├── scaler.pkl
    └── config.txt
```

### Сравнение scaler файлов

Используйте утилиту для сравнения:

```python
from utils.feature_validator import compare_scalers

compare_scalers(
    'workspace/prepared/scalers/feature_scaler_encoder_v1.pkl',
    'workspace/prepared/scalers/feature_scaler_encoder_v2.pkl'
)
```

## Экспорт документации по фичам

### Автоматический экспорт

После обучения модели автоматически создается документация по всем фичам:

- **JSON формат**: `workspace/models/checkpoints/{model_type}_model_features_documentation.json`
- **Markdown формат**: `workspace/models/checkpoints/{model_type}_model_features_documentation.md`

### Ручной экспорт

Для существующей модели:

```bash
# Экспорт для encoder модели
python export_features_doc.py --model-type encoder

# Экспорт для timeseries модели
python export_features_doc.py --model-type timeseries

# С указанием пути к scaler
python export_features_doc.py --scaler workspace/prepared/scalers/feature_scaler_encoder.pkl
```

### Что включает документация

Для каждого фича:
- **Название** и **описание**
- **Формула расчета** (если удалось извлечь из кода)
- **Категория** (ценовые, технические индикаторы, волатильность и т.д.)
- **Источник** (файл и функция, которая создает фичу)
- **Статистика** из обучающей выборки (mean, std)

### Преимущества

- **Не запутаетесь**: каждая модель имеет свою документацию
- **Легко передавать**: получатель видит, как рассчитываются фичи
- **Версионирование**: документация сохраняется вместе с моделью
- **Отслеживание изменений**: можно сравнить документации разных версий

## Следующие шаги

1. Проведите анализ корреляции на ваших данных
2. Удалите высококоррелированные фичи
3. Переобучите модель на очищенных данных
4. **Проверьте соответствие фичей** перед применением модели
5. Сравните результаты до и после оптимизации
6. Настройте мониторинг аномалий под ваши требования
7. **Ведите журнал экспериментов** с разными настройками
8. **Используйте документацию по фичам** для отслеживания изменений

