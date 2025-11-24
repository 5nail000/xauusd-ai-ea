# FX AI EA - Система генерации фичей для торгового робота

Проект для создания фичей (признаков) для обучения AI/ML моделей на данных Forex.

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

## Использование

Базовый пример использования:

```python
from data.mt5_data_loader import MT5DataLoader
from features.feature_engineering import FeatureEngineer
from datetime import datetime, timedelta

# Загрузка данных
loader = MT5DataLoader()
loader.connect()
df = loader.load_data('EURUSD', 'M1', 
                     start_date=datetime.now() - timedelta(days=30),
                     end_date=datetime.now())
loader.disconnect()

# Генерация фичей
engineer = FeatureEngineer()
df_features = engineer.create_features(df, add_targets=True)

# Создание последовательностей для Transformer
X, y = engineer.create_sequences(df_features, sequence_length=60)
```

Подробный пример см. в `example_usage.py`

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

## Следующие шаги

1. Обучение Transformer модели на созданных фичах
2. Валидация и тестирование модели
3. Интеграция с MetaTrader 5 для реальной торговли
4. Бэктестинг и оптимизация

## Лицензия

MIT

