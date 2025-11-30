"""
Конфигурация параметров для генерации фичей
"""
from dataclasses import dataclass
from typing import List

@dataclass
class FeatureConfig:
    """Конфигурация параметров индикаторов и фичей"""
    
    # Периоды для скользящих средних
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    
    # MACD параметры
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # RSI периоды
    rsi_periods: List[int] = None
    
    # ADX параметры
    adx_period: int = 14
    
    # Stochastic параметры
    stoch_k: int = 14
    stoch_d: int = 3
    stoch_smooth: int = 3
    
    # ATR периоды
    atr_periods: List[int] = None
    
    # Bollinger Bands параметры
    bb_period: int = 20
    bb_std: float = 2.0
    
    # CCI параметры
    cci_period: int = 20
    
    # Williams %R параметры
    williams_period: int = 14
    
    # Momentum параметры
    momentum_period: int = 10
    
    # ROC параметры
    roc_period: int = 12
    
    # Периоды для lag features
    lag_periods: List[int] = None
    
    # Rolling window для статистических фичей
    rolling_windows: List[int] = None
    
    # Мультитаймфреймы
    higher_timeframes: List[str] = None
    
    # Параметры для целевых переменных
    future_return_periods: List[int] = None
    
    # Параметры для тиковых данных
    tick_lookback_minutes: int = 1  # Количество минут тиковых данных для загрузки
    tick_candle_intervals: List[int] = None  # Интервалы для секундных свечей (в секундах)
    
    # Параметры для удаления коррелированных фичей
    remove_correlated_features: bool = False  # Удалять ли высококоррелированные фичи
    correlation_threshold: float = 0.95  # Порог корреляции для удаления
    
    # Параметры для уровней поддержки/сопротивления
    sr_lookback_window: int = 100  # Окно для поиска уровней (количество свечей)
    sr_extrema_window: int = 5  # Размер окна для поиска локальных экстремумов
    sr_min_touches: int = 2  # Минимальное количество касаний для формирования уровня (уменьшено с 3 до 2 для лучшего покрытия)
    sr_cluster_atr_multiplier: float = 2.0  # Множитель ATR для определения близости экстремумов
    
    # Параметры для Fibonacci
    fib_swing_window: int = 50  # Окно для поиска swing high/low
    fib_levels: List[float] = None  # Уровни Fibonacci (по умолчанию стандартные)
    
    # Параметры для Pivot Points
    use_pivot_points: bool = False  # Использовать ли Pivot Points
    pivot_type: str = 'classic'  # Тип pivot points ('classic', 'fibonacci', 'camarilla')
    
    def __post_init__(self):
        """Инициализация значений по умолчанию"""
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [5, 10, 20, 50, 100]
        if self.rsi_periods is None:
            self.rsi_periods = [6, 14, 21]
        if self.atr_periods is None:
            self.atr_periods = [6, 14, 21]
        if self.lag_periods is None:
            self.lag_periods = [1, 2, 3, 5, 10, 20]
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]
        if self.higher_timeframes is None:
            self.higher_timeframes = ['H1', 'H4', 'D1']
        if self.future_return_periods is None:
            self.future_return_periods = [1, 5, 10, 20, 60]
        if self.tick_candle_intervals is None:
            self.tick_candle_intervals = [1, 2, 3, 5, 10, 15, 20, 30, 45]
        if self.fib_levels is None:
            self.fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]

# Глобальный экземпляр конфигурации
default_config = FeatureConfig()

