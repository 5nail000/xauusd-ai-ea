"""
Модуль для генерации технических индикаторов
"""
import pandas as pd
import numpy as np
from typing import List, Optional
try:
    import pandas_ta as ta
except ImportError:
    print("Предупреждение: pandas_ta не установлен. Некоторые индикаторы могут быть недоступны.")
    ta = None

def add_trend_indicators(df: pd.DataFrame, sma_periods: List[int] = None, 
                        ema_periods: List[int] = None, macd_fast: int = 12,
                        macd_slow: int = 26, macd_signal: int = 9,
                        adx_period: int = 14) -> pd.DataFrame:
    """
    Добавляет индикаторы тренда
    
    Args:
        df: DataFrame с колонками open, high, low, close
        sma_periods: Периоды для SMA
        ema_periods: Периоды для EMA
        macd_fast: Быстрый период для MACD
        macd_slow: Медленный период для MACD
        macd_signal: Период сигнальной линии MACD
        adx_period: Период для ADX
    
    Returns:
        DataFrame с добавленными индикаторами
    """
    df = df.copy()
    
    if sma_periods is None:
        sma_periods = [5, 10, 20, 50, 100, 200]
    if ema_periods is None:
        ema_periods = [5, 10, 20, 50, 100]
    
    # SMA (Simple Moving Average)
    for period in sma_periods:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        # Отношение цены к SMA
        df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        # Расстояние от цены до SMA в процентах
        df[f'price_sma_distance_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
    
    # EMA (Exponential Moving Average)
    for period in ema_periods:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        # Отношение цены к EMA
        df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']
        # Расстояние от цены до EMA в процентах
        df[f'price_ema_distance_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
    
    # MACD
    if ta is not None:
        macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
        if macd is not None and not macd.empty:
            df['macd'] = macd[f'MACD_{macd_fast}_{macd_slow}_{macd_signal}']
            df['macd_signal'] = macd[f'MACDs_{macd_fast}_{macd_slow}_{macd_signal}']
            df['macd_histogram'] = macd[f'MACDh_{macd_fast}_{macd_slow}_{macd_signal}']
    else:
        # Ручной расчет MACD
        ema_fast = df['close'].ewm(span=macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=macd_slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=macd_signal, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # ADX (Average Directional Index)
    if ta is not None:
        adx = ta.adx(df['high'], df['low'], df['close'], length=adx_period)
        if adx is not None and not adx.empty:
            df['adx'] = adx[f'ADX_{adx_period}']
            df['adx_pos'] = adx[f'ADX_{adx_period}']
            df['adx_neg'] = adx[f'ADX_{adx_period}']
            if f'DMP_{adx_period}' in adx.columns:
                df['di_plus'] = adx[f'DMP_{adx_period}']
            if f'DMN_{adx_period}' in adx.columns:
                df['di_minus'] = adx[f'DMN_{adx_period}']
    else:
        # Упрощенный расчет ADX
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = calculate_true_range(df)
        atr = tr.rolling(window=adx_period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=adx_period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=adx_period).mean()
        df['di_plus'] = plus_di
        df['di_minus'] = minus_di
    
    # Parabolic SAR
    if ta is not None:
        psar = ta.psar(df['high'], df['low'], df['close'])
        if psar is not None and not psar.empty:
            psar_cols = [col for col in psar.columns if 'PSAR' in col]
            if psar_cols:
                df['psar'] = psar[psar_cols[0]]
                df['psar_signal'] = (df['close'] > df['psar']).astype(int)
    
    # Ichimoku Cloud компоненты
    if ta is not None:
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'])
        if ichimoku is not None and not ichimoku.empty:
            for col in ichimoku.columns:
                df[f'ichimoku_{col.lower()}'] = ichimoku[col]
    
    return df

def add_oscillators(df: pd.DataFrame, rsi_periods: List[int] = None,
                    stoch_k: int = 14, stoch_d: int = 3, stoch_smooth: int = 3,
                    cci_period: int = 20, williams_period: int = 14,
                    momentum_period: int = 10, roc_period: int = 12) -> pd.DataFrame:
    """
    Добавляет осцилляторы
    
    Args:
        df: DataFrame с колонками open, high, low, close
        rsi_periods: Периоды для RSI
        stoch_k: Период %K для Stochastic
        stoch_d: Период %D для Stochastic
        stoch_smooth: Период сглаживания для Stochastic
        cci_period: Период для CCI
        williams_period: Период для Williams %R
        momentum_period: Период для Momentum
        roc_period: Период для ROC
    
    Returns:
        DataFrame с добавленными осцилляторами
    """
    df = df.copy()
    
    if rsi_periods is None:
        rsi_periods = [14, 21]
    
    # RSI (Relative Strength Index)
    for period in rsi_periods:
        if ta is not None:
            rsi = ta.rsi(df['close'], length=period)
            if rsi is not None:
                df[f'rsi_{period}'] = rsi
        else:
            # Ручной расчет RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    if ta is not None:
        stoch = ta.stoch(df['high'], df['low'], df['close'], 
                        k=stoch_k, d=stoch_d, smooth_k=stoch_smooth)
        if stoch is not None and not stoch.empty:
            stoch_cols = [col for col in stoch.columns if 'STOCH' in col]
            if len(stoch_cols) >= 2:
                df['stoch_k'] = stoch[stoch_cols[0]]
                df['stoch_d'] = stoch[stoch_cols[1]]
    else:
        # Ручной расчет Stochastic
        lowest_low = df['low'].rolling(window=stoch_k).min()
        highest_high = df['high'].rolling(window=stoch_k).max()
        df['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        df['stoch_d'] = df['stoch_k'].rolling(window=stoch_d).mean()
    
    # CCI (Commodity Channel Index)
    if ta is not None:
        cci = ta.cci(df['high'], df['low'], df['close'], length=cci_period)
        if cci is not None:
            df['cci'] = cci
    else:
        # Ручной расчет CCI
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=cci_period).mean()
        mad = typical_price.rolling(window=cci_period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
    
    # Williams %R
    if ta is not None:
        willr = ta.willr(df['high'], df['low'], df['close'], length=williams_period)
        if willr is not None:
            df['williams_r'] = willr
    else:
        # Ручной расчет Williams %R
        highest_high = df['high'].rolling(window=williams_period).max()
        lowest_low = df['low'].rolling(window=williams_period).min()
        df['williams_r'] = -100 * (highest_high - df['close']) / (highest_high - lowest_low + 1e-10)
    
    # Momentum
    df['momentum'] = df['close'].diff(periods=momentum_period)
    df['momentum_pct'] = df['close'].pct_change(periods=momentum_period) * 100
    
    # ROC (Rate of Change)
    df['roc'] = ((df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period)) * 100
    
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

