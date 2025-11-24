"""
Модуль для распознавания свечных паттернов
"""
import pandas as pd
import numpy as np
from typing import Optional

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет признаки свечных паттернов
    
    Args:
        df: DataFrame с колонками open, high, low, close
    
    Returns:
        DataFrame с добавленными признаками паттернов
    """
    df = df.copy()
    
    # Базовые характеристики свечей
    body = abs(df['close'] - df['open'])
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
    total_range = df['high'] - df['low']
    
    # Doji (маленькое тело, большие тени)
    body_pct = body / (total_range + 1e-10)
    df['is_doji'] = (body_pct < 0.1).astype(int)
    
    # Hammer (маленькое тело вверху, длинная нижняя тень)
    is_bullish_hammer = (
        (lower_shadow > 2 * body) & 
        (upper_shadow < body) & 
        (df['close'] > df['open'])
    )
    is_bearish_hammer = (
        (lower_shadow > 2 * body) & 
        (upper_shadow < body) & 
        (df['close'] < df['open'])
    )
    df['is_hammer'] = (is_bullish_hammer | is_bearish_hammer).astype(int)
    df['is_bullish_hammer'] = is_bullish_hammer.astype(int)
    df['is_bearish_hammer'] = is_bearish_hammer.astype(int)
    
    # Shooting Star (маленькое тело внизу, длинная верхняя тень)
    df['is_shooting_star'] = (
        (upper_shadow > 2 * body) & 
        (lower_shadow < body) & 
        (df['close'] < df['open'])
    ).astype(int)
    
    # Marubozu (нет теней или очень маленькие)
    df['is_marubozu'] = (
        (upper_shadow < body * 0.1) & 
        (lower_shadow < body * 0.1) &
        (body > total_range * 0.9)
    ).astype(int)
    df['is_bullish_marubozu'] = (
        df['is_marubozu'] & 
        (df['close'] > df['open'])
    ).astype(int)
    df['is_bearish_marubozu'] = (
        df['is_marubozu'] & 
        (df['close'] < df['open'])
    ).astype(int)
    
    # Engulfing patterns
    prev_body = body.shift(1)
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    
    # Bullish Engulfing
    df['is_bullish_engulfing'] = (
        (df['close'] > df['open']) &  # Текущая свеча бычья
        (prev_close < prev_open) &  # Предыдущая свеча медвежья
        (df['open'] < prev_close) &  # Текущий open ниже предыдущего close
        (df['close'] > prev_open) &  # Текущий close выше предыдущего open
        (body > prev_body)  # Текущее тело больше предыдущего
    ).astype(int)
    
    # Bearish Engulfing
    df['is_bearish_engulfing'] = (
        (df['close'] < df['open']) &  # Текущая свеча медвежья
        (prev_close > prev_open) &  # Предыдущая свеча бычья
        (df['open'] > prev_close) &  # Текущий open выше предыдущего close
        (df['close'] < prev_open) &  # Текущий close ниже предыдущего open
        (body > prev_body)  # Текущее тело больше предыдущего
    ).astype(int)
    
    # Three Line Strike (три последовательные свечи одного направления)
    df['is_three_line_strike_bullish'] = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'].shift(2) > df['open'].shift(2)) &
        (df['close'] > df['close'].shift(1)) &
        (df['close'].shift(1) > df['close'].shift(2))
    ).astype(int)
    
    df['is_three_line_strike_bearish'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'].shift(2) < df['open'].shift(2)) &
        (df['close'] < df['close'].shift(1)) &
        (df['close'].shift(1) < df['close'].shift(2))
    ).astype(int)
    
    # Spinning Top (маленькое тело, большие тени с обеих сторон)
    df['is_spinning_top'] = (
        (body_pct < 0.3) &
        (upper_shadow > body) &
        (lower_shadow > body)
    ).astype(int)
    
    # Long-legged Doji (doji с очень длинными тенями)
    df['is_long_legged_doji'] = (
        df['is_doji'] &
        (upper_shadow > total_range * 0.4) &
        (lower_shadow > total_range * 0.4)
    ).astype(int)
    
    # Dragonfly Doji (doji с длинной нижней тенью)
    df['is_dragonfly_doji'] = (
        df['is_doji'] &
        (lower_shadow > total_range * 0.6) &
        (upper_shadow < total_range * 0.1)
    ).astype(int)
    
    # Gravestone Doji (doji с длинной верхней тенью)
    df['is_gravestone_doji'] = (
        df['is_doji'] &
        (upper_shadow > total_range * 0.6) &
        (lower_shadow < total_range * 0.1)
    ).astype(int)
    
    # Количество последовательных бычьих/медвежьих свечей
    df['consecutive_bullish'] = 0
    df['consecutive_bearish'] = 0
    
    bullish_mask = df['close'] > df['open']
    bearish_mask = df['close'] < df['open']
    
    # Подсчет последовательных бычьих свечей
    bullish_groups = (bullish_mask != bullish_mask.shift()).cumsum()
    df.loc[bullish_mask, 'consecutive_bullish'] = bullish_groups.groupby(bullish_groups).cumcount() + 1
    
    # Подсчет последовательных медвежьих свечей
    bearish_groups = (bearish_mask != bearish_mask.shift()).cumsum()
    df.loc[bearish_mask, 'consecutive_bearish'] = bearish_groups.groupby(bearish_groups).cumcount() + 1
    
    # Inside Bar (текущая свеча полностью внутри предыдущей)
    df['is_inside_bar'] = (
        (df['high'] < df['high'].shift(1)) &
        (df['low'] > df['low'].shift(1))
    ).astype(int)
    
    # Outside Bar (текущая свеча полностью охватывает предыдущую)
    df['is_outside_bar'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['low'] < df['low'].shift(1))
    ).astype(int)
    
    # Pin Bar (одна длинная тень, маленькое тело)
    df['is_pin_bar'] = (
        ((upper_shadow > 2 * body) | (lower_shadow > 2 * body)) &
        (body < total_range * 0.3)
    ).astype(int)
    df['is_pin_bar_bullish'] = (
        df['is_pin_bar'] &
        (lower_shadow > upper_shadow)
    ).astype(int)
    df['is_pin_bar_bearish'] = (
        df['is_pin_bar'] &
        (upper_shadow > lower_shadow)
    ).astype(int)
    
    return df

