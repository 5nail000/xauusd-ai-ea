# Тиковые фичи для минутного скальпера

## Описание

Модули для работы с тиковыми данными и создания секундных свечей для минутного скальпера.

## Модули

### `data/tick_data_loader.py`

Класс `TickDataLoader` для загрузки тиковых данных из MetaTrader 5:

- `load_ticks()` - загрузка тиков за указанный период
- `load_ticks_for_minute()` - загрузка тиков за последние N минут перед минутной свечой
- `load_ticks_batch()` - пакетная загрузка тиков для множества минутных свечей

### `features/tick_features.py`

Функции для генерации фичей из тиков:

- `create_second_candles_from_ticks()` - создание секундных свечей из тиков
- `add_tick_positioning_features()` - позиционирование цены относительно High/Low
- `add_tick_statistics_features()` - статистические фичи на основе тиков
- `aggregate_second_candles_features()` - агрегация фичей из секундных свечей
- `add_tick_features_to_minute_data()` - добавление тиковых фичей к минутным данным

## Секундные свечи

Создаются свечи для следующих интервалов:
- 1 секунда
- 2 секунды
- 3 секунды
- 5 секунд
- 10 секунд
- 15 секунд
- 20 секунд
- 30 секунд
- 45 секунд

### Особенности расчета

- **High** рассчитывается по **bid** (максимальный bid за период)
- **Low** рассчитывается по **ask** (минимальный ask за период)
- **Close** использует последний **bid**

## Типовые фичи

Для каждого интервала секундных свечей создаются:

### Позиционирование
- `close_position_in_range` - позиция close в диапазоне High-Low (0-1)
- `distance_to_high_pct` - расстояние до High в процентах
- `distance_to_low_pct` - расстояние до Low в процентах
- `distance_to_high_norm` - нормализованное расстояние до High
- `distance_to_low_norm` - нормализованное расстояние до Low

### Спред
- `spread_mean` - средний спред bid/ask
- `spread_max` - максимальный спред
- `spread_min` - минимальный спред
- `spread_std` - стандартное отклонение спреда

### Тиковый объем
- `tick_count_sum` - общее количество тиков
- `tick_count_mean` - среднее количество тиков
- `tick_count_max` - максимальное количество тиков

### Скорость и волатильность
- `tick_rate_mean` - средняя скорость тиков (тики/сек)
- `tick_rate_max` - максимальная скорость тиков
- `tick_volatility_mean` - средняя волатильность тиков
- `bid_velocity` - скорость изменения bid
- `ask_velocity` - скорость изменения ask

### Распределение тиков
- `bid_up_ticks` - количество тиков с ростом bid
- `bid_down_ticks` - количество тиков с падением bid
- `bid_up_down_ratio` - соотношение роста/падения bid

## Использование

```python
from data.mt5_data_loader import MT5DataLoader
from data.tick_data_loader import TickDataLoader
from features.feature_engineering import FeatureEngineer

# Загрузка минутных данных
loader = MT5DataLoader()
loader.connect()
df = loader.load_data('EURUSD', 'M1', start_date, end_date)

# Загрузка тиковых данных
tick_loader = TickDataLoader()
ticks_data = tick_loader.load_ticks_batch(
    symbol='EURUSD',
    minute_times=df.index,
    lookback_minutes=2  # Последние 2 минуты
)

# Генерация фичей с тиковыми данными
engineer = FeatureEngineer()
df_features = engineer.create_features(
    df,
    ticks_data=ticks_data,
    add_targets=True
)
```

## Конфигурация

Параметры настраиваются в `config/feature_config.py`:

```python
tick_lookback_minutes: int = 2  # Количество минут тиков для загрузки
tick_candle_intervals: List[int] = [1, 2, 3, 5, 10, 15, 20, 30, 45]  # Интервалы свечей
```

## Обновления индикаторов

Добавлены периоды:
- **RSI**: 6, 14, 21 (ранее было только 14, 21)
- **ATR**: 6, 14, 21 (ранее было только 14, 21)

