# FX AI EA - AI Trading System для Forex (XAUUSD)

Полнофункциональная система для обучения Transformer моделей и торговли на Forex с использованием тиковых данных и технических индикаторов.

## Структура проекта

```
011_FX_AI_EA/
├── data/                    # Модули для загрузки данных
│   ├── mt5_data_loader.py   # Загрузка данных из MetaTrader 5
│   └── __init__.py
├── features/                # Модули генерации фичей
│   ├── price_features.py           # Базовые ценовые фичи
│   ├── technical_indicators.py     # Технические индикаторы
│   ├── volatility_features.py      # Фичи волатильности
│   ├── time_features.py            # Временные фичи
│   ├── pattern_features.py         # Свечные паттерны
│   ├── multitimeframe_features.py  # Мультитаймфреймовые фичи
│   ├── statistical_features.py     # Статистические фичи
│   ├── volume_features.py          # Объемные фичи
│   ├── feature_engineering.py      # Главный модуль
│   └── __init__.py
├── models/                  # Модели машинного обучения (будут добавлены)
├── utils/                   # Вспомогательные утилиты
│   ├── normalization.py    # Нормализация данных
│   └── __init__.py
├── config/                  # Конфигурационные файлы
│   ├── feature_config.py    # Конфигурация параметров фичей
│   └── __init__.py
├── requirements.txt         # Зависимости проекта
├── example_usage.py         # Пример использования
└── README.md
```

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. (Опционально) Установите pandas-ta для расширенной поддержки технических индикаторов:
```bash
# Для Python 3.11 попробуйте установить последнюю версию:
pip install pandas-ta

# Или для Python 3.10 и ниже:
pip install pandas-ta==0.3.14b
```

**Примечание:** pandas-ta опциональна. Если она не установлена, код автоматически использует ручной расчет индикаторов (fallback режим).

3. Убедитесь, что MetaTrader 5 установлен и запущен (для загрузки данных)

## Быстрый старт

### 1. Полный цикл (рекомендуется)

```bash
# Полный цикл: подготовка данных → обучение → бэктестинг
python full_pipeline.py --months 12

# С параметрами
python full_pipeline.py --months 12 --encoder-only --epochs 50

# Справка по параметрам
python full_pipeline.py --help
```

### 2. Или по отдельности

#### Подготовка данных

```bash
# Подготовка данных на 12 месяцев
python prepare_gold_data.py --months 12

# Справка по параметрам
python prepare_gold_data.py --help
```

#### Обучение моделей

```bash
# Обучение обеих моделей (encoder и timeseries)
python train_all_models.py

# Только encoder модель
python train_all_models.py --encoder-only

# Справка по параметрам
python train_all_models.py --help
```

#### Бэктестинг

```bash
python backtest_strategy.py
```

## Основные скрипты

- **`full_pipeline.py`** ⭐ - **Единый скрипт для полного цикла** (подготовка → обучение → бэктестинг)
- **`prepare_gold_data.py`** - подготовка данных с аргументами командной строки
- **`train_all_models.py`** - обучение всех моделей автоматически (с экспортом документации)
- **`train_model.py`** - обучение одной модели (с экспортом документации)
- **`backtest_strategy.py`** - бэктестинг торговой стратегии
- **`test_pipeline.py`** - тестовый цикл на небольшом объеме данных
- **`validate_features.py`** - валидация соответствия фичей между данными и моделью
- **`export_features_doc.py`** - экспорт документации по фичам для существующей модели
- **`analyze_feature_correlation.py`** - анализ корреляции фичей
- **`export_features_doc.py`** - экспорт документации по фичам для существующей модели

## Программное использование

```python
from data.gold_data_prep import GoldDataPreparator
from models.model_factory import create_model
from trading.backtester import Backtester

# Подготовка данных
preparator = GoldDataPreparator(training_months=12)
df = preparator.prepare_full_dataset(symbol='XAUUSD', months=12)

# Обучение модели (см. train_model.py)
# ...

# Бэктестинг
backtester = Backtester(
    model_path='models/checkpoints/encoder_model.pth',
    scaler_path='models/feature_scaler_encoder.pkl'
)
results = backtester.backtest(test_df)
```

## Типы фичей

### 1. Базовые ценовые фичи
- OHLC данные
- Returns (процентные изменения)
- Log returns
- Spreads (High-Low, Open-Close)
- Тени свечей

### 2. Технические индикаторы
- **Тренд**: SMA, EMA, MACD, ADX, Parabolic SAR, Ichimoku
- **Осцилляторы**: RSI, Stochastic, CCI, Williams %R, Momentum, ROC

### 3. Волатильность
- ATR (Average True Range)
- Bollinger Bands
- Historical Volatility
- Parkinson и Garman-Klass estimators

### 4. Временные фичи
- Час, день недели, месяц
- Циклическое кодирование (sin/cos)
- Торговые сессии (London, New York, Tokyo)
- Перекрытия сессий

### 5. Свечные паттерны
- Doji, Hammer, Shooting Star
- Engulfing patterns
- Marubozu
- Inside/Outside bars

### 6. Мультитаймфреймовые фичи
- Данные с H1, H4, D1 таймфреймов
- Отношения цен между таймфреймами
- Тренды на старших таймфреймах

### 7. Статистические фичи
- Z-score нормализация
- Rolling statistics (mean, std, min, max, median)
- Skewness и Kurtosis
- Autocorrelation

### 8. Объемные фичи (если доступны)
- OBV (On Balance Volume)
- Volume Moving Average
- VWAP
- Chaikin Money Flow

## Конфигурация

Параметры индикаторов можно настроить в `config/feature_config.py`:

```python
from config.feature_config import FeatureConfig

config = FeatureConfig(
    sma_periods=[5, 10, 20, 50, 100, 200],
    rsi_periods=[14, 21],
    # ... другие параметры
)
```

## Нормализация

Модуль `utils/normalization.py` предоставляет:
- `FeatureScaler` - класс для масштабирования фичей
- Методы: StandardScaler, MinMaxScaler, RobustScaler
- Сохранение/загрузка scaler для использования в продакшене

## Кэширование и сохранение прогресса

Система автоматически сохраняет прогресс на каждом этапе:

- **Генерация фичей**: промежуточные результаты сохраняются после каждого этапа
- **Загрузка тиков**: прогресс батчевой загрузки сохраняется автоматически
- **Финальный датасет**: сохраняется в кэш для быстрой загрузки при повторном запуске

При прерывании процесса можно продолжить с последнего сохраненного этапа.

## Мониторинг обучения

После обучения автоматически создаются:

- **История обучения** (CSV): все метрики по эпохам
- **Графики обучения** (PNG): кривые loss, accuracy, learning rate
- **Анализ переобученности**: автоматический анализ и рекомендации
- **Confusion Matrix**: визуализация ошибок классификации

## Документация

Вся документация находится в папке `docs/`:

1. **[01_README.md](docs/01_README.md)** - главный файл проекта (вы здесь)
2. **[02_INSTALL.md](docs/02_INSTALL.md)** - установка и настройка
3. **[03_DATA_PREPARATION.md](docs/03_DATA_PREPARATION.md)** - подготовка данных
4. **[04_TRAINING_GUIDE.md](docs/04_TRAINING_GUIDE.md)** - обучение моделей
5. **[05_MONITORING_GUIDE.md](docs/05_MONITORING_GUIDE.md)** - мониторинг и анализ обучения
6. **[06_BACKTESTING_GUIDE.md](docs/06_BACKTESTING_GUIDE.md)** - бэктестинг стратегии
7. **[07_MODEL_ARCHITECTURE.md](docs/07_MODEL_ARCHITECTURE.md)** - архитектуры моделей
8. **[08_TICK_FEATURES.md](docs/08_TICK_FEATURES.md)** - тиковые фичи
9. **[09_TICK_CACHE_GUIDE.md](docs/09_TICK_CACHE_GUIDE.md)** - кэширование тиков
10. **[10_TRADING_MODEL_PLAN.md](docs/10_TRADING_MODEL_PLAN.md)** - план проекта и статус

## Лицензия

MIT

