# Подготовка данных для обучения модели

## Обзор

Модули для подготовки данных по золоту (XAUUSD) для обучения Transformer модели классификации торговых сигналов.

## Классификация сигналов

Модель классифицирует три типа сигналов:

### 1. Пробой (Breakout) - класс 1
- **Условие**: будущая доходность > 50 пунктов в одном направлении
- **Характеристика**: сильное движение в одну сторону без значительного разворота
- **Использование**: вход в позицию по направлению пробоя

### 2. Отскок (Bounce) - класс 2
- **Условие**: разворот после движения, доходность меняет знак, движение > 30 пунктов в обратную сторону
- **Характеристика**: начальное движение в одну сторону, затем разворот
- **Использование**: вход в позицию против начального движения

### 3. Неопределенность (Uncertainty) - класс 0
- **Условие**: движение < 50 пунктов или отсутствие четкого направления
- **Характеристика**: слабое движение, боковой тренд
- **Использование**: воздержание от торговли

## Модули

### `data/target_generator.py`

Класс `TargetGenerator` для генерации целевых переменных:

```python
from data.target_generator import TargetGenerator

generator = TargetGenerator(
    breakout_threshold=50.0,  # Порог для пробоя в пунктах
    bounce_threshold=30.0,    # Порог для отскока в пунктах
    lookahead_periods=60      # Количество периодов вперед
)

# Генерация целевых переменных
df_with_targets = generator.generate_targets(df, price_column='close')
```

**Методы:**
- `generate_targets()` - генерация всех целевых переменных
- `get_class_distribution()` - анализ распределения классов
- `balance_classes()` - балансировка классов (если нужно)

### `data/gold_data_prep.py`

Класс `GoldDataPreparator` для подготовки данных по золоту:

```python
from data.gold_data_prep import GoldDataPreparator

preparator = GoldDataPreparator(
    training_months=6  # Количество месяцев данных
)

# Подготовка полного датасета
df = preparator.prepare_full_dataset(
    symbol='XAUUSD',
    months=6,
    load_ticks=True,
    load_higher_tf=True
)
```

**Методы:**
- `load_gold_data()` - загрузка минутных данных
- `load_higher_timeframes()` - загрузка старших таймфреймов
- `load_tick_data()` - загрузка тиковых данных
- `prepare_full_dataset()` - полная подготовка датасета
- `save_prepared_data()` / `load_prepared_data()` - сохранение/загрузка

### `data/data_splitter.py`

Класс `DataSplitter` для разделения данных:

```python
from data.data_splitter import DataSplitter

splitter = DataSplitter(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    temporal_split=True  # Временное разделение (без перемешивания)
)

train_df, val_df, test_df = splitter.split(df, target_column='signal_class')
```

**Методы:**
- `split()` - разделение данных
- `get_class_distribution()` - анализ распределения классов

## Использование

### Быстрый старт

Запустите скрипт подготовки данных:

```bash
python prepare_gold_data.py
```

Скрипт:
1. Загрузит данные по золоту (6 месяцев по умолчанию)
2. Загрузит тиковые данные и старшие таймфреймы
3. Сгенерирует все фичи
4. Создаст целевые переменные
5. Разделит данные на train/val/test
6. Сохранит результаты в файлы

### Программное использование

```python
from data.gold_data_prep import GoldDataPreparator
from data.data_splitter import DataSplitter

# Подготовка данных
preparator = GoldDataPreparator(training_months=6)
df = preparator.prepare_full_dataset(
    symbol='XAUUSD',
    months=6,
    load_ticks=True,
    load_higher_tf=True
)

# Разделение данных
splitter = DataSplitter(temporal_split=True)
train_df, val_df, test_df = splitter.split(df)

# Сохранение
train_df.to_csv('data/gold_train.csv')
val_df.to_csv('data/gold_val.csv')
test_df.to_csv('data/gold_test.csv')
```

## Параметры

### Количество месяцев данных

Можно указать любое количество месяцев:

```python
# 6 месяцев (по умолчанию)
preparator = GoldDataPreparator(training_months=6)

# 1 год
preparator = GoldDataPreparator(training_months=12)

# 3 месяца (для быстрого тестирования)
preparator = GoldDataPreparator(training_months=3)
```

### Пороги для классификации

Можно настроить пороги для пробоя и отскока:

```python
from data.target_generator import TargetGenerator

generator = TargetGenerator(
    breakout_threshold=50.0,  # Порог пробоя (пункты)
    bounce_threshold=30.0,    # Порог отскока (пункты)
    lookahead_periods=60     # Период анализа (минуты)
)
```

## Выходные файлы

После подготовки данных создаются файлы:

- `data/gold_data_6months.csv` - полный датасет
- `data/gold_train.csv` - обучающая выборка (70%)
- `data/gold_val.csv` - валидационная выборка (15%)
- `data/gold_test.csv` - тестовая выборка (15%)

## Структура данных

Подготовленный DataFrame содержит:

- **Фичи**: все сгенерированные фичи (ценовые, технические, тиковые и т.д.)
- **Целевые переменные**:
  - `signal_class`: числовой класс (0, 1, 2)
  - `signal_class_name`: название класса ('uncertainty', 'breakout', 'bounce')
  - `future_return_N`: доходности на разных периодах
  - `max_future_return`: максимальная будущая доходность

## Следующие шаги

После подготовки данных можно:
1. Загрузить данные для обучения модели
2. Создать последовательности для Transformer
3. Обучить модель
4. Оценить результаты

