"""
Модуль для генерации мультитаймфреймовых фичей
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def add_multitimeframe_features(df_current: pd.DataFrame, 
                                df_higher: Dict[str, pd.DataFrame],
                                timeframes: List[str] = None) -> pd.DataFrame:
    """
    Добавляет фичи с старших таймфреймов
    
    Args:
        df_current: DataFrame текущего таймфрейма
        df_higher: Словарь {timeframe: DataFrame} с данными старших таймфреймов
        timeframes: Список таймфреймов для использования
    
    Returns:
        DataFrame с добавленными мультитаймфреймовыми фичами
    """
    df = df_current.copy()
    
    if timeframes is None:
        timeframes = ['H1', 'H4', 'D1']
    
    for tf in timeframes:
        if tf not in df_higher:
            continue
        
        df_tf = df_higher[tf]
        
        # Ресэмплинг текущего таймфрейма к старшему для синхронизации
        # Используем последнее значение (close) для агрегации
        df_tf_resampled = df_tf.reindex(df.index, method='ffill')
        
        # Базовые ценовые фичи со старшего таймфрейма
        df[f'{tf}_close'] = df_tf_resampled['close']
        df[f'{tf}_open'] = df_tf_resampled['open']
        df[f'{tf}_high'] = df_tf_resampled['high']
        df[f'{tf}_low'] = df_tf_resampled['low']
        
        # Отношение текущей цены к цене на старшем таймфрейме
        df[f'price_to_{tf}_close'] = df['close'] / (df[f'{tf}_close'] + 1e-10)
        df[f'price_to_{tf}_high'] = df['close'] / (df[f'{tf}_high'] + 1e-10)
        df[f'price_to_{tf}_low'] = df['close'] / (df[f'{tf}_low'] + 1e-10)
        
        # Расстояние от текущей цены до цен старшего таймфрейма
        df[f'distance_to_{tf}_close'] = (df['close'] - df[f'{tf}_close']) / df[f'{tf}_close']
        df[f'distance_to_{tf}_high'] = (df['close'] - df[f'{tf}_high']) / df[f'{tf}_high']
        df[f'distance_to_{tf}_low'] = (df['close'] - df[f'{tf}_low']) / df[f'{tf}_low']
        
        # Позиция текущей цены в диапазоне старшего таймфрейма
        tf_range = df[f'{tf}_high'] - df[f'{tf}_low']
        df[f'position_in_{tf}_range'] = (
            (df['close'] - df[f'{tf}_low']) / (tf_range + 1e-10)
        )
        
        # Тренд на старшем таймфрейме
        if len(df_tf) > 1:
            # Простой тренд: цена выше/ниже предыдущей
            tf_close_prev = df_tf_resampled['close'].shift(1)
            df[f'{tf}_trend'] = (df[f'{tf}_close'] > tf_close_prev).astype(int)
            df[f'{tf}_trend_strength'] = (df[f'{tf}_close'] - tf_close_prev) / (tf_close_prev + 1e-10)
        
        # SMA на старшем таймфрейме
        if len(df_tf) >= 20:
            sma_20 = df_tf['close'].rolling(window=20).mean()
            sma_20_resampled = sma_20.reindex(df.index, method='ffill')
            df[f'{tf}_sma_20'] = sma_20_resampled
            df[f'price_to_{tf}_sma_20'] = df['close'] / (sma_20_resampled + 1e-10)
        
        if len(df_tf) >= 50:
            sma_50 = df_tf['close'].rolling(window=50).mean()
            sma_50_resampled = sma_50.reindex(df.index, method='ffill')
            df[f'{tf}_sma_50'] = sma_50_resampled
            df[f'price_to_{tf}_sma_50'] = df['close'] / (sma_50_resampled + 1e-10)
        
        # RSI на старшем таймфрейме
        if len(df_tf) >= 14:
            delta = df_tf['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            rsi_resampled = rsi.reindex(df.index, method='ffill')
            df[f'{tf}_rsi'] = rsi_resampled
        
        # ATR на старшем таймфрейме
        if len(df_tf) >= 14:
            high_low = df_tf['high'] - df_tf['low']
            high_close = abs(df_tf['high'] - df_tf['close'].shift(1))
            low_close = abs(df_tf['low'] - df_tf['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            atr_resampled = atr.reindex(df.index, method='ffill')
            df[f'{tf}_atr'] = atr_resampled
            df[f'{tf}_atr_pct'] = (atr_resampled / df[f'{tf}_close']) * 100
    
    return df

def resample_to_higher_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Ресэмплинг данных на старший таймфрейм
    
    Args:
        df: DataFrame с данными
        timeframe: Целевой таймфрейм ('H1', 'H4', 'D1')
    
    Returns:
        Ресэмплированный DataFrame
    """
    timeframe_map = {
        'H1': '1H',
        'H4': '4H',
        'D1': '1D',
        'M15': '15T',
        'M30': '30T'
    }
    
    if timeframe not in timeframe_map:
        raise ValueError(f"Неподдерживаемый таймфрейм: {timeframe}")
    
    resample_rule = timeframe_map[timeframe]
    
    df_resampled = pd.DataFrame()
    df_resampled['open'] = df['open'].resample(resample_rule).first()
    df_resampled['high'] = df['high'].resample(resample_rule).max()
    df_resampled['low'] = df['low'].resample(resample_rule).min()
    df_resampled['close'] = df['close'].resample(resample_rule).last()
    
    if 'volume' in df.columns:
        df_resampled['volume'] = df['volume'].resample(resample_rule).sum()
    
    return df_resampled

