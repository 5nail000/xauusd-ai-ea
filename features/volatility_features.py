"""
Модуль для генерации фичей волатильности
"""
import pandas as pd
import numpy as np
from typing import List, Optional
try:
    import pandas_ta as ta
except ImportError:
    ta = None

def add_volatility_features(df: pd.DataFrame, atr_periods: List[int] = None,
                           bb_period: int = 20, bb_std: float = 2.0,
                           volatility_windows: List[int] = None) -> pd.DataFrame:
    """
    Добавляет фичи волатильности
    
    Args:
        df: DataFrame с колонками open, high, low, close
        atr_periods: Периоды для ATR
        bb_period: Период для Bollinger Bands
        bb_std: Количество стандартных отклонений для Bollinger Bands
        volatility_windows: Окна для расчета исторической волатильности
    
    Returns:
        DataFrame с добавленными фичами волатильности
    """
    df = df.copy()
    
    if atr_periods is None:
        atr_periods = [14, 21]
    if volatility_windows is None:
        volatility_windows = [5, 10, 20, 50]
    
    # True Range
    tr = calculate_true_range(df)
    df['true_range'] = tr
    
    # ATR (Average True Range)
    for period in atr_periods:
        df[f'atr_{period}'] = tr.rolling(window=period).mean()
        # Нормализованный ATR (в процентах от цены)
        df[f'atr_{period}_pct'] = (df[f'atr_{period}'] / df['close']) * 100
        # Отношение текущего TR к ATR
        df[f'tr_to_atr_{period}'] = tr / (df[f'atr_{period}'] + 1e-10)
    
    # Bollinger Bands
    if ta is not None:
        bb = ta.bbands(df['close'], length=bb_period, std=bb_std)
        if bb is not None and not bb.empty:
            bb_cols = [col for col in bb.columns if 'BB' in col]
            if len(bb_cols) >= 3:
                df['bb_upper'] = bb[bb_cols[0]]
                df['bb_middle'] = bb[bb_cols[1]]
                df['bb_lower'] = bb[bb_cols[2]]
    else:
        # Ручной расчет Bollinger Bands
        sma = df['close'].rolling(window=bb_period).mean()
        std = df['close'].rolling(window=bb_period).std()
        df['bb_middle'] = sma
        df['bb_upper'] = sma + (std * bb_std)
        df['bb_lower'] = sma - (std * bb_std)
    
    # Ширина Bollinger Bands
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_width_pct'] = (df['bb_width'] / df['bb_middle']) * 100
    
    # Позиция цены в Bollinger Bands (0 = нижняя полоса, 1 = верхняя полоса)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
    
    # Расстояние от цены до полос Bollinger Bands
    df['bb_distance_upper'] = (df['close'] - df['bb_upper']) / df['close']
    df['bb_distance_lower'] = (df['bb_lower'] - df['close']) / df['close']
    
    # Standard Deviation (скользящее стандартное отклонение)
    for window in volatility_windows:
        df[f'std_{window}'] = df['close'].rolling(window=window).std()
        df[f'std_{window}_pct'] = (df[f'std_{window}'] / df['close']) * 100
    
    # Historical Volatility (историческая волатильность)
    for window in volatility_windows:
        returns = df['close'].pct_change()
        df[f'historical_vol_{window}'] = returns.rolling(window=window).std() * np.sqrt(window)
        # Годовая волатильность (для минутных данных умножаем на sqrt(525600))
        df[f'historical_vol_annual_{window}'] = df[f'historical_vol_{window}'] * np.sqrt(525600)
    
    # Parkinson Volatility Estimator (использует high и low)
    for window in volatility_windows:
        hl_ratio = np.log(df['high'] / df['low'])
        df[f'parkinson_vol_{window}'] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(window=window).mean()
        )
    
    # Garman-Klass Volatility Estimator
    for window in volatility_windows:
        hl = np.log(df['high'] / df['low']) ** 2
        co = np.log(df['close'] / df['open']) ** 2
        df[f'gk_vol_{window}'] = np.sqrt(
            0.5 * hl.rolling(window=window).mean() - 
            (2 * np.log(2) - 1) * co.rolling(window=window).mean()
        )
    
    # Realized Volatility (реализованная волатильность на основе returns)
    for window in volatility_windows:
        returns = df['close'].pct_change()
        df[f'realized_vol_{window}'] = returns.rolling(window=window).std()
    
    # Volatility Ratio (отношение краткосрочной к долгосрочной волатильности)
    if len(volatility_windows) >= 2:
        short_window = volatility_windows[0]
        long_window = volatility_windows[-1]
        df['volatility_ratio'] = (
            df[f'historical_vol_{short_window}'] / 
            (df[f'historical_vol_{long_window}'] + 1e-10)
        )
    
    return df

def calculate_true_range(df: pd.DataFrame) -> pd.Series:
    """
    Вычисляет True Range
    
    Args:
        df: DataFrame с колонками high, low, close
    
    Returns:
        Series с True Range
    """
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr

