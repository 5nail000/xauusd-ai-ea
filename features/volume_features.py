"""
Модуль для генерации объемных фичей (если доступны данные об объеме)
"""
import pandas as pd
import numpy as np
from typing import Optional

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет объемные фичи
    
    Args:
        df: DataFrame с колонкой volume (тиковый объем)
    
    Returns:
        DataFrame с добавленными объемными фичами
    """
    df = df.copy()
    
    if 'volume' not in df.columns:
        print("Предупреждение: колонка 'volume' не найдена. Объемные фичи не будут добавлены.")
        return df
    
    # OBV (On Balance Volume)
    df['obv'] = 0.0
    price_change = df['close'].diff()
    
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1, df.columns.get_loc('obv')] + df['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1, df.columns.get_loc('obv')] - df['volume'].iloc[i]
        else:
            df.iloc[i, df.columns.get_loc('obv')] = df.iloc[i-1, df.columns.get_loc('obv')]
    
    # Volume Moving Average
    for period in [5, 10, 20, 50]:
        df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_ma_{period}'] + 1e-10)
    
    # Volume Rate of Change
    for period in [5, 10, 20]:
        df[f'volume_roc_{period}'] = df['volume'].pct_change(periods=period) * 100
    
    # Price-Volume Trend
    df['pvt'] = 0.0
    for i in range(1, len(df)):
        pct_change = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
        df.iloc[i, df.columns.get_loc('pvt')] = (
            df.iloc[i-1, df.columns.get_loc('pvt')] + 
            df['volume'].iloc[i] * pct_change
        )
    
    # Volume Weighted Average Price (VWAP) - для каждого дня
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['price_to_vwap'] = df['close'] / (df['vwap'] + 1e-10)
    
    # Volume Profile (высокий/низкий объем)
    volume_mean = df['volume'].rolling(window=20).mean()
    volume_std = df['volume'].rolling(window=20).std()
    df['is_high_volume'] = (df['volume'] > volume_mean + volume_std).astype(int)
    df['is_low_volume'] = (df['volume'] < volume_mean - volume_std).astype(int)
    
    # Volume-Price Trend
    df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
    
    # Accumulation/Distribution Line
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
    df['ad_line'] = (clv * df['volume']).cumsum()
    
    # Chaikin Money Flow
    for period in [14, 21]:
        mf_volume = clv * df['volume']
        df[f'cmf_{period}'] = mf_volume.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    
    # Volume Oscillator
    short_ma = df['volume'].rolling(window=5).mean()
    long_ma = df['volume'].rolling(window=10).mean()
    df['volume_oscillator'] = ((short_ma - long_ma) / (long_ma + 1e-10)) * 100
    
    return df

