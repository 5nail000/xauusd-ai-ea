"""
Модуль для генерации временных фичей
"""
import pandas as pd
import numpy as np
from typing import Optional
from datetime import time

def add_time_features(df: pd.DataFrame, market_open_time: time = time(8, 0),
                     market_close_time: time = time(20, 0)) -> pd.DataFrame:
    """
    Добавляет временные фичи
    
    Args:
        df: DataFrame с индексом datetime
        market_open_time: Время открытия рынка (по умолчанию 8:00)
        market_close_time: Время закрытия рынка (по умолчанию 20:00)
    
    Returns:
        DataFrame с добавленными временными фичами
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Индекс DataFrame должен быть DatetimeIndex")
    
    # Базовые временные фичи
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek  # 0 = Monday, 6 = Sunday
    df['day_of_month'] = df.index.day
    df['week_of_year'] = df.index.isocalendar().week
    df['month'] = df.index.month
    df['minute_of_hour'] = df.index.minute
    
    # Циклическое кодирование для часов (sin/cos)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Циклическое кодирование для дня недели
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Циклическое кодирование для месяца
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Циклическое кодирование для минут в часе
    df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_hour'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_hour'] / 60)
    
    # Торговые сессии
    df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_new_york_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['is_tokyo_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    
    # Перекрытие сессий
    df['is_london_ny_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    
    # Время с начала торгового дня
    df['time_since_market_open'] = (
        (df.index.hour - market_open_time.hour) * 60 + 
        (df.index.minute - market_open_time.minute)
    )
    df['time_since_market_open'] = df['time_since_market_open'].clip(lower=0)
    
    # Время до закрытия рынка
    df['time_to_market_close'] = (
        (market_close_time.hour - df.index.hour) * 60 + 
        (market_close_time.minute - df.index.minute)
    )
    df['time_to_market_close'] = df['time_to_market_close'].clip(lower=0)
    
    # Рынок открыт/закрыт
    market_open_hour = market_open_time.hour
    market_close_hour = market_close_time.hour
    
    if market_close_hour > market_open_hour:
        # Обычный случай: рынок открыт в течение дня
        df['is_market_open'] = (
            (df['hour'] >= market_open_hour) & 
            (df['hour'] < market_close_hour)
        ).astype(int)
    else:
        # Рынок открыт через полночь
        df['is_market_open'] = (
            (df['hour'] >= market_open_hour) | 
            (df['hour'] < market_close_hour)
        ).astype(int)
    
    # Выходные дни
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Первый/последний час торгового дня
    df['is_first_hour'] = (df['hour'] == market_open_hour).astype(int)
    df['is_last_hour'] = (df['hour'] == (market_close_hour - 1)).astype(int)
    
    # Время дня (утро, день, вечер, ночь)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    
    return df

