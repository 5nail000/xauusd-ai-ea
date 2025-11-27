# Подготовка данных для обучения модели

## Обзор

Модули для подготовки данных по золоту (XAUUSD) для обучения Transformer модели классификации торговых сигналов.

## Классификация сигналов

Модель классифицирует 5 типов сигналов:

### 0. Неопределенность (Uncertainty)
- **Условие**: движение < порогов или отсутствие четкого направления
- **Характеристика**: слабое движение, боковой тренд
- **Использование**: воздержание от торговли

### 1. Пробой вверх (Breakout Up) - BUY
- **Условие**: будущая доходность > 200 пунктов вверх
- **Характеристика**: сильное движение вверх без значительного разворота
- **Использование**: вход в длинную позицию

### 2. Пробой вниз (Breakout Down) - SELL
- **Условие**: будущая доходность < -200 пунктов (движение вниз)
- **Характеристика**: сильное движение вниз без значительного разворота
- **Использование**: вход в короткую позицию

### 3. Отскок вверх (Bounce Up) - BUY после падения
- **Условие**: разворот после падения, доходность меняет знак, движение > 150 пунктов вверх
- **Характеристика**: начальное движение вниз, затем разворот вверх
- **Использование**: вход в длинную позицию после подтверждения разворота (отложенный вход)

### 4. Отскок вниз (Bounce Down) - SELL после роста
- **Условие**: разворот после роста, доходность меняет знак, движение < -150 пунктов (вниз)
- **Характеристика**: начальное движение вверх, затем разворот вниз
- **Использование**: вход в короткую позицию после подтверждения разворота (отложенный вход)

## Модули

### `data/target_generator.py`

Класс `TargetGenerator` для генерации целевых переменных:

```python
from data.target_generator import TargetGenerator

generator = TargetGenerator(
    breakout_threshold=450.0,  # Порог для пробоя в пунктах (по умолчанию 450)
    bounce_threshold=350.0,    # Порог для отскока в пунктах (по умолчанию 350)
    lookahead_periods=60       # Количество периодов вперед (минуты)
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
- `load_gold_data()` - загрузка минутных данных (автоматически создает из тиков, если данных недостаточно)
- `load_higher_timeframes()` - загрузка старших таймфреймов
- `load_tick_data()` - загрузка тиковых данных
- `prepare_full_dataset()` - полная подготовка датасета
- `save_prepared_data()` / `load_prepared_data()` - сохранение/загрузка

**Особенность**: Если минутных данных недостаточно (брокер хранит только 3-6 месяцев), система автоматически:
1. Загружает тики за недостающие периоды
2. Создает минутные свечи из тиков
3. Объединяет данные из MT5 и созданные из тиков

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
# Базовое использование (6 месяцев)
python prepare_gold_data.py

# 12 месяцев
python prepare_gold_data.py --months 12

# Указать количество дней (приоритет над --months)
python prepare_gold_data.py --days 30
python prepare_gold_data.py -d 7

# Справка по всем параметрам
python prepare_gold_data.py --help
```

Скрипт:
1. Загрузит данные по золоту (6 месяцев по умолчанию, можно указать через `--months` или `--days`)
2. Загрузит тиковые данные и старшие таймфреймы (можно отключить через `--no-ticks` или `--no-higher-tf`)
3. Сгенерирует все фичи (с автоматическим сохранением прогресса)
4. Создаст целевые переменные
5. Разделит данные на train/val/test
6. Сохранит результаты в файлы

### Параметры командной строки

```bash
# Количество месяцев
python prepare_gold_data.py -m 12
python prepare_gold_data.py --months 12

# Количество дней (приоритет над --months)
python prepare_gold_data.py --days 30
python prepare_gold_data.py -d 7

# Без тиков
python prepare_gold_data.py -m 12 --no-ticks

# Без старших таймфреймов
python prepare_gold_data.py -m 12 --no-higher-tf

# Принудительная регенерация (игнорировать кэш)
python prepare_gold_data.py -m 12 --force

# Не использовать кэш вообще
python prepare_gold_data.py -m 12 --no-cache

# Автоматически загружать сохраненные данные (не спрашивать)
python prepare_gold_data.py -m 12 --no-ask

# Комбинация параметров
python prepare_gold_data.py -m 12 --no-ask --force
```

**Все параметры:**
- `-m, --months` - количество месяцев данных (по умолчанию: 6)
- `-d, --days` - количество дней данных (приоритет над --months)
- `--symbol` - торговый символ (по умолчанию: XAUUSD)
- `--no-ticks` - не загружать тиковые данные
- `--no-higher-tf` - не загружать старшие таймфреймы
- `--force` - принудительно регенерировать данные (игнорировать кэш)
- `--no-cache` - не использовать кэш (не сохранять и не загружать)
- `--no-ask` - не спрашивать при наличии сохраненных данных (автоматически загружать)
- `--offline` - режим offline - работа только с кэшированными данными без подключения к MT5

### Режим offline (без подключения к MT5)

Для работы на Linux машинах или без доступа к MT5 можно использовать режим offline:

```bash
# Подготовка данных в offline режиме (только из кэша)
python prepare_gold_data.py --offline --days 30

# Полный цикл в offline режиме
python full_pipeline.py --offline --days 30
```

**Требования для offline режима:**
- Тики должны быть загружены в `workspace/raw_data/ticks/`
- Данные должны покрывать требуемый период
- MT5 не требуется (можно запускать на Linux)

**Как это работает:**
1. Загружает тики из кэша `workspace/raw_data/ticks/`
2. Создает минутные свечи из тиков
3. Создает старшие таймфреймы через агрегацию из минутных данных
4. Генерирует фичи и целевые переменные

**Преимущества:**
- ✅ Работа на Linux без MT5
- ✅ Использование только кэшированных данных
- ✅ Старшие таймфреймы создаются автоматически из минутных данных
- ✅ Полная совместимость с обычным режимом

### Программное использование

```python
from data.gold_data_prep import GoldDataPreparator
from data.data_splitter import DataSplitter

# Подготовка данных (обычный режим)
preparator = GoldDataPreparator(training_months=6)
df = preparator.prepare_full_dataset(
    symbol='XAUUSD',
    months=6,
    load_ticks=True,
    load_higher_tf=True
)

# Подготовка данных (offline режим)
preparator_offline = GoldDataPreparator(training_months=6, offline_mode=True)
df = preparator_offline.prepare_full_dataset(
    symbol='XAUUSD',
    days=30,
    load_ticks=True,
    load_higher_tf=True
)

# Разделение данных
splitter = DataSplitter(temporal_split=True)
train_df, val_df, test_df = splitter.split(df)

# Сохранение
train_df.to_csv('workspace/prepared/features/gold_train.csv')
val_df.to_csv('workspace/prepared/features/gold_val.csv')
test_df.to_csv('workspace/prepared/features/gold_test.csv')
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
    breakout_threshold=450.0,  # Порог пробоя (пункты, по умолчанию 450)
    bounce_threshold=350.0,    # Порог отскока (пункты, по умолчанию 350)
    lookahead_periods=60       # Период анализа (минуты)
)
```

## Выходные файлы

После подготовки данных создаются файлы:

- `workspace/prepared/features/gold_data_{months}months.csv` - полный датасет
- `workspace/prepared/features/gold_train.csv` - обучающая выборка (70%)
- `workspace/prepared/features/gold_val.csv` - валидационная выборка (15%)
- `workspace/prepared/features/gold_test.csv` - тестовая выборка (15%)

## Структура данных

Подготовленный DataFrame содержит:

- **Фичи**: все сгенерированные фичи (ценовые, технические, тиковые и т.д.)
- **Целевые переменные**:
  - `signal_class`: числовой класс (0, 1, 2)
  - `signal_class_name`: название класса ('uncertainty', 'breakout', 'bounce')
  - `future_return_N`: доходности на разных периодах
  - `max_future_return`: максимальная будущая доходность

## Кэширование и сохранение прогресса

### Автоматическое сохранение

Система автоматически сохраняет прогресс на каждом этапе:

1. **Промежуточные фичи** (`workspace/raw_data/cache/`):
   - Сохраняются после каждого этапа генерации фичей
   - При прерывании процесс продолжается с последнего этапа
   - Автоматически очищаются после успешного завершения

2. **Тиковые данные** (`workspace/raw_data/ticks/`):
   - Кэшируются по дням в формате `{symbol}_{YYYYMMDD}.pkl`
   - При загрузке тиков для минутных свечей система **сначала проверяет кэш** для каждой свечи
   - Загружает из MT5 **только недостающие** тики (оптимизированная логика)
   - Показывает статистику использования кэша (сколько данных из кэша vs из MT5)
   - Прогресс батчевой загрузки сохраняется
   - При прерывании продолжается с последней обработанной свечи
   - **Важно**: По умолчанию система не загружает весь диапазон кэша заранее (1.5 года), 
     а загружает тики по требованию. Это ускоряет работу, если данные уже в кэше.

3. **Финальный датасет** (`workspace/raw_data/cache/`):
   - Сохраняется в кэш с уникальным именем на основе параметров
   - При повторном запуске с теми же параметрами загружается из кэша
   - Имя файла: `{symbol}_{months}m_{date}_{ticks_flag}_{tf_flag}.pkl`

### Возобновление после прерывания

Если процесс был прерван (Ctrl+C, ошибка, перезагрузка):

1. **Генерация фичей**: автоматически продолжается с последнего сохраненного этапа
2. **Загрузка тиков**: автоматически продолжается с последней обработанной свечи
3. **Финальный датасет**: если уже был создан, загружается из кэша

### Управление кэшем

```python
# Принудительная регенерация (игнорировать кэш)
df = preparator.prepare_full_dataset(
    months=12,
    force_regenerate=True
)

# Не использовать кэш вообще
df = preparator.prepare_full_dataset(
    months=12,
    use_cache=False
)

# Автоматически загружать сохраненные данные (не спрашивать)
df = preparator.prepare_full_dataset(
    months=12,
    ask_on_existing=False
)
```

## Оптимизация фичей

### Удаление высококоррелированных фичей

Можно включить автоматическое удаление высококоррелированных фичей:

```python
from config.feature_config import FeatureConfig

config = FeatureConfig(
    remove_correlated_features=True,  # Включить удаление
    correlation_threshold=0.95        # Порог корреляции
)
```

Или использовать отдельный скрипт для анализа:

```bash
python analyze_feature_correlation.py --remove --threshold 0.95
```

Подробнее см. [11_FEATURE_OPTIMIZATION.md](11_FEATURE_OPTIMIZATION.md)

## Визуализация данных обучения

Для визуализации подготовленных данных с сигналами и точками закрытия используется скрипт `utils/visualize_training_data.py`:

```bash
# Визуализация последних 5 дней (по умолчанию)
python utils/visualize_training_data.py

# Визуализация последних 10 дней
python utils/visualize_training_data.py --days 10

# Визуализация с конкретной даты
python utils/visualize_training_data.py --start-date 2024-01-01 --days 7

# Сохранение графика в файл
python utils/visualize_training_data.py --output workspace/results/training_visualization.png

# Показать также неопределенность (класс 0)
python utils/visualize_training_data.py --show-uncertainty
```

**Что отображается на графике:**

- **Свечной график**: минутные свечи (open, high, low, close)
- **Маркеры сигналов**: 
  - Зеленый треугольник вверх (^) - пробой вверх (класс 1) и отскок вверх (класс 3)
  - Красный треугольник вниз (v) - пробой вниз (класс 2) и отскок вниз (класс 4)
- **Линии к точкам закрытия**: пунктирные линии от сигнала к точке закрытия
- **Точки закрытия**: крестики (x) показывают, где должен закрыться сигнал согласно разметке

**Параметры:**

- `--input`: путь к файлу с данными (по умолчанию: `workspace/prepared/features/gold_train.csv`)
- `--days`: количество дней для отображения (по умолчанию: 5)
- `--start-date`: начальная дата в формате YYYY-MM-DD
- `--output`: путь для сохранения графика (если не указан, показывается на экране)
- `--show-uncertainty`: показывать неопределенность (класс 0) на графике

## Следующие шаги

После подготовки данных можно:
1. Проанализировать корреляцию фичей (опционально)
2. Загрузить данные для обучения модели
3. Обучить модель(и) через `train_all_models.py`
4. Провести бэктестинг стратегии
5. Оценить результаты

