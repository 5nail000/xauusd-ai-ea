"""
Главный модуль для генерации всех фичей
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, time

from config.feature_config import FeatureConfig, default_config
from features.price_features import add_price_features, add_lag_features
from features.technical_indicators import add_trend_indicators, add_oscillators
from features.volatility_features import add_volatility_features
from features.time_features import add_time_features
from features.pattern_features import add_candle_patterns
from features.multitimeframe_features import add_multitimeframe_features, resample_to_higher_timeframe
from features.statistical_features import add_statistical_features, add_momentum_features
from features.volume_features import add_volume_features
from features.tick_features import add_tick_features_to_minute_data

class FeatureEngineer:
    """
    Класс для генерации всех фичей из сырых данных
    """
    
    def __init__(self, config: FeatureConfig = None):
        """
        Args:
            config: Конфигурация параметров фичей
        """
        self.config = config if config is not None else default_config
    
    def create_features(self, df: pd.DataFrame, 
                       higher_timeframes_data: Optional[Dict[str, pd.DataFrame]] = None,
                       ticks_data: Optional[Dict[datetime, pd.DataFrame]] = None,
                       add_targets: bool = True) -> pd.DataFrame:
        """
        Создает все фичи из сырых данных
        
        Args:
            df: DataFrame с колонками open, high, low, close, volume (опционально)
            higher_timeframes_data: Словарь {timeframe: DataFrame} с данными старших таймфреймов
            ticks_data: Словарь {minute_time: ticks_df} с тиковыми данными для каждой минутной свечи
            add_targets: Добавлять ли целевые переменные
        
        Returns:
            DataFrame со всеми фичами
        """
        print("Генерация фичей...")
        
        # 1. Базовые ценовые фичи
        print("  - Базовые ценовые фичи...")
        df = add_price_features(df)
        
        # 2. Технические индикаторы - тренд
        print("  - Индикаторы тренда...")
        df = add_trend_indicators(
            df,
            sma_periods=self.config.sma_periods,
            ema_periods=self.config.ema_periods,
            macd_fast=self.config.macd_fast,
            macd_slow=self.config.macd_slow,
            macd_signal=self.config.macd_signal,
            adx_period=self.config.adx_period
        )
        
        # 3. Осцилляторы
        print("  - Осцилляторы...")
        df = add_oscillators(
            df,
            rsi_periods=self.config.rsi_periods,
            stoch_k=self.config.stoch_k,
            stoch_d=self.config.stoch_d,
            stoch_smooth=self.config.stoch_smooth,
            cci_period=self.config.cci_period,
            williams_period=self.config.williams_period,
            momentum_period=self.config.momentum_period,
            roc_period=self.config.roc_period
        )
        
        # 4. Волатильность
        print("  - Фичи волатильности...")
        df = add_volatility_features(
            df,
            atr_periods=self.config.atr_periods,
            bb_period=self.config.bb_period,
            bb_std=self.config.bb_std,
            volatility_windows=self.config.rolling_windows
        )
        
        # 5. Временные фичи
        print("  - Временные фичи...")
        df = add_time_features(df)
        
        # 6. Свечные паттерны
        print("  - Свечные паттерны...")
        df = add_candle_patterns(df)
        
        # 7. Объемные фичи (если доступны)
        if 'volume' in df.columns:
            print("  - Объемные фичи...")
            df = add_volume_features(df)
        
        # 8. Статистические фичи
        print("  - Статистические фичи...")
        df = add_statistical_features(
            df,
            columns=['close', 'returns'],
            rolling_windows=self.config.rolling_windows
        )
        
        # 9. Фичи момента
        print("  - Фичи момента...")
        df = add_momentum_features(
            df,
            columns=['close'],
            periods=[5, 10, 20, 50]
        )
        
        # 10. Lag features
        print("  - Lag features...")
        df = add_lag_features(
            df,
            columns=['close', 'returns', 'volume'] if 'volume' in df.columns else ['close', 'returns'],
            periods=self.config.lag_periods
        )
        
        # 11. Мультитаймфреймовые фичи
        if higher_timeframes_data is not None:
            print("  - Мультитаймфреймовые фичи...")
            df = add_multitimeframe_features(
                df,
                higher_timeframes_data,
                timeframes=self.config.higher_timeframes
            )
        
        # 12. Тиковые фичи (секундные свечи)
        if ticks_data is not None and len(ticks_data) > 0:
            print("  - Тиковые фичи (секундные свечи)...")
            df = add_tick_features_to_minute_data(
                df,
                ticks_data,
                intervals=self.config.tick_candle_intervals
            )
        
        # 13. Целевые переменные
        if add_targets:
            print("  - Целевые переменные...")
            df = self._add_target_variables(df)
        
        # Удаление строк с NaN (после расчета индикаторов)
        initial_len = len(df)
        df = df.dropna()
        print(f"\nУдалено {initial_len - len(df)} строк с NaN из {initial_len} всего")
        
        print(f"\nИтого создано {len(df.columns)} фичей")
        print(f"Размер данных: {len(df)} строк")
        
        return df
    
    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет целевые переменные для обучения
        
        Args:
            df: DataFrame с фичами
        
        Returns:
            DataFrame с добавленными целевыми переменными
        """
        df = df.copy()
        
        # Future returns (доходность через N периодов)
        for period in self.config.future_return_periods:
            df[f'future_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
        
        # Direction (направление движения)
        for period in [1, 5, 10, 20]:
            future_price = df['close'].shift(-period)
            df[f'direction_{period}'] = np.where(
                future_price > df['close'], 1,
                np.where(future_price < df['close'], -1, 0)
            )
        
        # Volatility target (волатильность в будущем)
        for period in [5, 10, 20]:
            future_returns = df['close'].pct_change(period).shift(-period)
            df[f'future_volatility_{period}'] = future_returns.rolling(window=period).std()
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60,
                        target_column: str = 'direction_1',
                        feature_columns: Optional[List[str]] = None) -> tuple:
        """
        Создает последовательности для Transformer модели
        
        Args:
            df: DataFrame с фичами
            sequence_length: Длина последовательности
            target_column: Колонка с целевой переменной
            feature_columns: Список колонок для использования (если None, все кроме целевых)
        
        Returns:
            Tuple (X, y) где X - массив последовательностей, y - массив целевых значений
        """
        if feature_columns is None:
            # Исключаем целевые переменные и временные метки
            exclude_patterns = ['target', 'label', 'direction', 'future_return', 'future_volatility']
            feature_columns = [
                col for col in df.columns 
                if not any(pattern in col.lower() for pattern in exclude_patterns)
                and col != target_column
            ]
        
        # Удаляем строки с NaN в целевой переменной
        df_clean = df.dropna(subset=[target_column])
        
        X = []
        y = []
        
        for i in range(sequence_length, len(df_clean)):
            # Последовательность фичей
            sequence = df_clean[feature_columns].iloc[i-sequence_length:i].values
            X.append(sequence)
            
            # Целевое значение
            y.append(df_clean[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """
        Возвращает список всех фичей (исключая целевые переменные)
        
        Args:
            df: DataFrame с фичами
        
        Returns:
            Список названий фичей
        """
        exclude_patterns = ['target', 'label', 'direction', 'future_return', 'future_volatility']
        feature_columns = [
            col for col in df.columns 
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]
        return feature_columns

