"""
Модуль для генерации статистических фичей
"""
import pandas as pd
import numpy as np
import warnings
from typing import List, Optional

# Подавляем предупреждения о фрагментации DataFrame (это не критично для работы)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
from scipy import stats

def add_statistical_features(df: pd.DataFrame, columns: List[str] = None,
                            rolling_windows: List[int] = None) -> pd.DataFrame:
    """
    Добавляет статистические фичи
    
    Args:
        df: DataFrame
        columns: Список колонок для расчета статистики (если None, используется 'close')
        rolling_windows: Окна для скользящих статистик
    
    Returns:
        DataFrame с добавленными статистическими фичами
    """
    df = df.copy()
    
    if columns is None:
        columns = ['close', 'returns']
    if rolling_windows is None:
        rolling_windows = [5, 10, 20, 50]
    
    # Фильтруем только существующие колонки
    columns = [col for col in columns if col in df.columns]
    
    for col in columns:
        for window in rolling_windows:
            if len(df) < window:
                continue
            
            # Rolling mean
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
            
            # Rolling std
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
            
            # Z-score (нормализация)
            mean = df[col].rolling(window=window).mean()
            std = df[col].rolling(window=window).std()
            df[f'{col}_zscore_{window}'] = (df[col] - mean) / (std + 1e-10)
            
            # Rolling min/max
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
            
            # Позиция текущего значения в rolling window
            rolling_min = df[f'{col}_rolling_min_{window}']
            rolling_max = df[f'{col}_rolling_max_{window}']
            rolling_range = rolling_max - rolling_min
            df[f'{col}_position_in_range_{window}'] = (
                (df[col] - rolling_min) / (rolling_range + 1e-10)
            )
            
            # Расстояние до min/max
            df[f'{col}_distance_to_min_{window}'] = (
                (df[col] - rolling_min) / (rolling_min + 1e-10)
            )
            df[f'{col}_distance_to_max_{window}'] = (
                (rolling_max - df[col]) / (rolling_max + 1e-10)
            )
            
            # Rolling median
            df[f'{col}_rolling_median_{window}'] = df[col].rolling(window=window).median()
            
            # Rolling quantiles
            df[f'{col}_rolling_q25_{window}'] = df[col].rolling(window=window).quantile(0.25)
            df[f'{col}_rolling_q75_{window}'] = df[col].rolling(window=window).quantile(0.75)
            df[f'{col}_rolling_iqr_{window}'] = (
                df[f'{col}_rolling_q75_{window}'] - df[f'{col}_rolling_q25_{window}']
            )
        
        # Skewness (асимметрия)
        for window in rolling_windows:
            if len(df) < window:
                continue
            df[f'{col}_skewness_{window}'] = df[col].rolling(window=window).skew()
        
        # Kurtosis (эксцесс)
        for window in rolling_windows:
            if len(df) < window:
                continue
            df[f'{col}_kurtosis_{window}'] = df[col].rolling(window=window).kurt()
        
        # Autocorrelation
        for window in rolling_windows:
            if len(df) < window * 2:
                continue
            autocorr_values = []
            for i in range(len(df)):
                if i < window:
                    autocorr_values.append(np.nan)
                else:
                    series = df[col].iloc[i-window+1:i+1]
                    if len(series) > 1 and series.std() > 0:
                        autocorr = series.autocorr(lag=1)
                        autocorr_values.append(autocorr if not np.isnan(autocorr) else 0)
                    else:
                        autocorr_values.append(0)
            df[f'{col}_autocorr_{window}'] = autocorr_values
    
    # Cross-correlation между колонками
    if len(columns) >= 2:
        for window in rolling_windows:
            if len(df) < window:
                continue
            col1, col2 = columns[0], columns[1]
            corr_values = []
            for i in range(len(df)):
                if i < window:
                    corr_values.append(np.nan)
                else:
                    series1 = df[col1].iloc[i-window+1:i+1]
                    series2 = df[col2].iloc[i-window+1:i+1]
                    if len(series1) > 1 and series1.std() > 0 and series2.std() > 0:
                        corr = series1.corr(series2)
                        corr_values.append(corr if not np.isnan(corr) else 0)
                    else:
                        corr_values.append(0)
            df[f'{col1}_{col2}_corr_{window}'] = corr_values
    
    # Percentile rank (процентный ранг текущего значения)
    for col in columns:
        for window in rolling_windows:
            if len(df) < window:
                continue
            percentile_ranks = []
            for i in range(len(df)):
                if i < window:
                    percentile_ranks.append(np.nan)
                else:
                    window_data = df[col].iloc[i-window+1:i+1]
                    current_value = df[col].iloc[i]
                    percentile_rank = stats.percentileofscore(window_data, current_value) / 100.0
                    percentile_ranks.append(percentile_rank)
            df[f'{col}_percentile_rank_{window}'] = percentile_ranks
    
    return df

def add_momentum_features(df: pd.DataFrame, columns: List[str] = None,
                         periods: List[int] = None) -> pd.DataFrame:
    """
    Добавляет фичи момента
    
    Args:
        df: DataFrame
        columns: Колонки для расчета
        periods: Периоды для расчета
    
    Returns:
        DataFrame с добавленными фичами момента
    """
    df = df.copy()
    
    if columns is None:
        columns = ['close']
    if periods is None:
        periods = [5, 10, 20, 50]
    
    columns = [col for col in columns if col in df.columns]
    
    for col in columns:
        for period in periods:
            # Rate of Change
            df[f'{col}_roc_{period}'] = df[col].pct_change(periods=period) * 100
            
            # Momentum
            df[f'{col}_momentum_{period}'] = df[col] - df[col].shift(period)
            
            # Relative Strength
            gains = df[col].diff().where(df[col].diff() > 0, 0).rolling(window=period).sum()
            losses = -df[col].diff().where(df[col].diff() < 0, 0).rolling(window=period).sum()
            df[f'{col}_rs_{period}'] = gains / (losses + 1e-10)
    
    return df

