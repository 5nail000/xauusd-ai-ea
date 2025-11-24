"""
Модуль для генерации базовых ценовых фичей
"""
import pandas as pd
import numpy as np
from typing import Optional

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет базовые ценовые фичи к DataFrame
    
    Args:
        df: DataFrame с колонками open, high, low, close
    
    Returns:
        DataFrame с добавленными фичами
    """
    df = df.copy()
    
    # Returns (процентные изменения цены)
    df['returns'] = df['close'].pct_change()
    
    # Log returns
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Абсолютные изменения цены
    df['price_change'] = df['close'].diff()
    df['price_change_open'] = df['open'].diff()
    df['price_change_high'] = df['high'].diff()
    df['price_change_low'] = df['low'].diff()
    
    # High-Low spread
    df['hl_spread'] = df['high'] - df['low']
    df['hl_spread_pct'] = df['hl_spread'] / df['close']
    
    # Open-Close spread
    df['oc_spread'] = abs(df['close'] - df['open'])
    df['oc_spread_pct'] = df['oc_spread'] / df['close']
    
    # Body size (размер тела свечи)
    df['body_size'] = abs(df['close'] - df['open'])
    df['body_size_pct'] = df['body_size'] / df['close']
    
    # Upper shadow (верхняя тень)
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['upper_shadow_pct'] = df['upper_shadow'] / df['close']
    
    # Lower shadow (нижняя тень)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['lower_shadow_pct'] = df['lower_shadow'] / df['close']
    
    # Total range
    df['total_range'] = df['high'] - df['low']
    df['total_range_pct'] = df['total_range'] / df['close']
    
    # Цена относительно диапазона свечи
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Направление свечи
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['close'] < df['open']).astype(int)
    df['is_doji'] = (abs(df['close'] - df['open']) / df['close'] < 0.0001).astype(int)
    
    return df

def add_lag_features(df: pd.DataFrame, columns: list, periods: list) -> pd.DataFrame:
    """
    Добавляет lag features (значения на предыдущих шагах)
    
    Args:
        df: DataFrame
        columns: Список колонок для создания lag features
        periods: Список периодов для lag (например, [1, 2, 3, 5, 10, 20])
    
    Returns:
        DataFrame с добавленными lag features
    """
    df = df.copy()
    
    for col in columns:
        for period in periods:
            df[f'{col}_lag_{period}'] = df[col].shift(period)
    
    return df

