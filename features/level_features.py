"""
Модуль для генерации фичей уровней поддержки/сопротивления и Fibonacci
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple


def find_local_extrema(series: pd.Series, window: int = 5, mode: str = 'min') -> pd.Series:
    """
    Находит локальные экстремумы в серии
    
    Args:
        series: Временной ряд
        window: Размер окна для поиска экстремумов
        mode: 'min' для минимумов, 'max' для максимумов
    
    Returns:
        Boolean Series, где True означает локальный экстремум
    """
    if mode == 'min':
        # Локальный минимум: значение меньше всех соседей в окне
        return series.rolling(window=window*2+1, center=True).min() == series
    else:
        # Локальный максимум: значение больше всех соседей в окне
        return series.rolling(window=window*2+1, center=True).max() == series


def cluster_extrema(extrema_values: np.ndarray, atr: float, min_touches: int = 3) -> List[List[float]]:
    """
    Группирует близкие экстремумы в кластеры (зоны)
    
    Args:
        extrema_values: Массив значений экстремумов
        atr: ATR для определения "близости"
        min_touches: Минимальное количество касаний для кластера
    
    Returns:
        Список кластеров (каждый кластер - список значений)
    """
    if len(extrema_values) < min_touches:
        return []
    
    # Сортируем значения
    sorted_values = np.sort(extrema_values)
    
    clusters = []
    current_cluster = [sorted_values[0]]
    
    for val in sorted_values[1:]:
        # Если значение близко к последнему в кластере (в пределах 2*ATR)
        if val - current_cluster[-1] < 2 * atr:
            current_cluster.append(val)
        else:
            # Сохраняем кластер, если в нем достаточно касаний
            if len(current_cluster) >= min_touches:
                clusters.append(current_cluster)
            current_cluster = [val]
    
    # Добавляем последний кластер
    if len(current_cluster) >= min_touches:
        clusters.append(current_cluster)
    
    return clusters


def add_support_resistance_features(df: pd.DataFrame,
                                   lookback_window: int = 100,
                                   extrema_window: int = 5,
                                   min_touches: int = 3,
                                   cluster_atr_multiplier: float = 2.0) -> pd.DataFrame:
    """
    Добавляет фичи уровней поддержки и сопротивления
    
    Идея:
    - Находит локальные минимумы (support) и максимумы (resistance)
    - Группирует близкие экстремумы в кластеры (зоны)
    - Использует центр и ширину зоны
    - Вычисляет относительное расстояние цены до зоны
    
    Args:
        df: DataFrame с колонками open, high, low, close
        lookback_window: Окно для поиска уровней (количество свечей назад)
        extrema_window: Размер окна для поиска локальных экстремумов
        min_touches: Минимальное количество касаний для формирования уровня
        cluster_atr_multiplier: Множитель ATR для определения близости экстремумов
    
    Returns:
        DataFrame с добавленными фичами уровней
    """
    df = df.copy()
    
    # Проверяем наличие ATR (если нет, используем приближение)
    if 'atr_14' in df.columns:
        atr = df['atr_14']
    else:
        # Приблизительный ATR как 0.1% от цены
        atr = df['close'] * 0.001
    
    # 1. Находим локальные экстремумы
    local_mins_mask = find_local_extrema(df['low'], window=extrema_window, mode='min')
    local_maxs_mask = find_local_extrema(df['high'], window=extrema_window, mode='max')
    
    local_mins = df['low'][local_mins_mask]
    local_maxs = df['high'][local_maxs_mask]
    
    # 2. Для каждого момента времени находим ближайшие уровни
    support_levels = []
    resistance_levels = []
    support_widths = []
    resistance_widths = []
    support_strengths = []
    resistance_strengths = []
    
    for i in range(len(df)):
        if i < lookback_window:
            support_levels.append(np.nan)
            resistance_levels.append(np.nan)
            support_widths.append(np.nan)
            resistance_widths.append(np.nan)
            support_strengths.append(0)
            resistance_strengths.append(0)
            continue
        
        # Берем экстремумы за последние lookback_window свечей
        window_start = max(0, i - lookback_window)
        window_mins = local_mins.iloc[window_start:i]
        window_maxs = local_maxs.iloc[window_start:i]
        
        current_atr = atr.iloc[i]
        
        # Обработка поддержек (минимумы)
        if len(window_mins) >= min_touches:
            clusters = cluster_extrema(
                window_mins.values,
                current_atr * cluster_atr_multiplier,
                min_touches
            )
            
            if clusters:
                # Берем самый сильный кластер (с наибольшим количеством касаний)
                strongest_cluster = max(clusters, key=len)
                support_center = np.median(strongest_cluster)
                support_width = np.std(strongest_cluster) if len(strongest_cluster) > 1 else current_atr * 0.5
                support_strength = len(strongest_cluster)
                
                support_levels.append(support_center)
                support_widths.append(support_width)
                support_strengths.append(support_strength)
            else:
                support_levels.append(np.nan)
                support_widths.append(np.nan)
                support_strengths.append(0)
        else:
            support_levels.append(np.nan)
            support_widths.append(np.nan)
            support_strengths.append(0)
        
        # Обработка сопротивлений (максимумы)
        if len(window_maxs) >= min_touches:
            clusters = cluster_extrema(
                window_maxs.values,
                current_atr * cluster_atr_multiplier,
                min_touches
            )
            
            if clusters:
                # Берем самый сильный кластер
                strongest_cluster = max(clusters, key=len)
                resistance_center = np.median(strongest_cluster)
                resistance_width = np.std(strongest_cluster) if len(strongest_cluster) > 1 else current_atr * 0.5
                resistance_strength = len(strongest_cluster)
                
                resistance_levels.append(resistance_center)
                resistance_widths.append(resistance_width)
                resistance_strengths.append(resistance_strength)
            else:
                resistance_levels.append(np.nan)
                resistance_widths.append(np.nan)
                resistance_strengths.append(0)
        else:
            resistance_levels.append(np.nan)
            resistance_widths.append(np.nan)
            resistance_strengths.append(0)
    
    # 3. Сохраняем уровни
    df['support_level'] = support_levels
    df['resistance_level'] = resistance_levels
    df['support_width'] = support_widths
    df['resistance_width'] = resistance_widths
    df['support_strength'] = support_strengths
    df['resistance_strength'] = resistance_strengths
    
    # 4. Вычисляем ОТНОСИТЕЛЬНЫЕ расстояния
    # Расстояние до поддержки в сигмах (стандартизация!)
    df['distance_to_support_sigma'] = (
        (df['close'] - df['support_level']) / (df['support_width'] + 1e-10)
    )
    
    # Расстояние до сопротивления в сигмах
    df['distance_to_resistance_sigma'] = (
        (df['resistance_level'] - df['close']) / (df['resistance_width'] + 1e-10)
    )
    
    # Расстояние в единицах ATR (более понятно для модели)
    df['distance_to_support_atr'] = (
        (df['close'] - df['support_level']) / (atr + 1e-10)
    )
    df['distance_to_resistance_atr'] = (
        (df['resistance_level'] - df['close']) / (atr + 1e-10)
    )
    
    # Процентное расстояние (относительно текущей цены)
    df['distance_to_support_pct'] = (
        (df['close'] - df['support_level']) / (df['close'] + 1e-10) * 100
    )
    df['distance_to_resistance_pct'] = (
        (df['resistance_level'] - df['close']) / (df['close'] + 1e-10) * 100
    )
    
    # Бинарные признаки: цена в зоне поддержки/сопротивления?
    df['in_support_zone'] = (abs(df['distance_to_support_sigma']) < 1.0).astype(int)
    df['in_resistance_zone'] = (abs(df['distance_to_resistance_sigma']) < 1.0).astype(int)
    
    # Близость к уровню (чем ближе, тем выше значение, экспоненциальный спад)
    df['proximity_to_support'] = np.exp(-abs(df['distance_to_support_sigma']) / 2.0)
    df['proximity_to_resistance'] = np.exp(-abs(df['distance_to_resistance_sigma']) / 2.0)
    
    # Нормализуем близость через ATR (более стабильно)
    df['proximity_to_support_atr'] = np.exp(-abs(df['distance_to_support_atr']) / 2.0)
    df['proximity_to_resistance_atr'] = np.exp(-abs(df['distance_to_resistance_atr']) / 2.0)
    
    # Отношение цены к уровню (для понимания выше/ниже)
    df['price_to_support_ratio'] = df['close'] / (df['support_level'] + 1e-10)
    df['price_to_resistance_ratio'] = df['close'] / (df['resistance_level'] + 1e-10)
    
    return df


def add_fibonacci_features(df: pd.DataFrame,
                           swing_window: int = 50,
                           fib_levels: List[float] = None) -> pd.DataFrame:
    """
    Вычисляет уровни Fibonacci Retracement от последнего swing high/low
    
    Идея:
    - Находит последний значимый swing high (для отката вниз)
    - Находит последний значимый swing low (для отката вверх)
    - Вычисляет уровни Fib как процент от диапазона swing
    - Использует ОТНОСИТЕЛЬНЫЕ расстояния, а не абсолютные
    
    Args:
        df: DataFrame с колонками open, high, low, close
        swing_window: Окно для поиска swing high/low
        fib_levels: Список уровней Fibonacci (по умолчанию стандартные)
    
    Returns:
        DataFrame с добавленными Fibonacci фичами
    """
    df = df.copy()
    
    if fib_levels is None:
        # Стандартные уровни Fibonacci
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # Проверяем наличие ATR
    if 'atr_14' in df.columns:
        atr = df['atr_14']
    else:
        atr = df['close'] * 0.001
    
    # Находим swing highs и lows
    swing_highs = []
    swing_lows = []
    
    for i in range(len(df)):
        if i < swing_window:
            swing_highs.append(np.nan)
            swing_lows.append(np.nan)
            continue
        
        window_df = df.iloc[i-swing_window:i]
        swing_highs.append(window_df['high'].max())
        swing_lows.append(window_df['low'].min())
    
    df['swing_high'] = swing_highs
    df['swing_low'] = swing_lows
    df['swing_range'] = df['swing_high'] - df['swing_low']
    
    # Для каждого уровня Fib вычисляем:
    for fib_level in fib_levels:
        fib_name = f'fib_{int(fib_level*1000)}'
        
        # Уровень Fib от swing low (для восходящего тренда)
        fib_price_from_low = df['swing_low'] + df['swing_range'] * fib_level
        
        # Сохраняем абсолютное значение уровня
        df[f'{fib_name}_price'] = fib_price_from_low
        
        # Расстояние до уровня Fib (относительно swing range)
        df[f'{fib_name}_distance_pct'] = (
            (df['close'] - fib_price_from_low) / (df['swing_range'] + 1e-10) * 100
        )
        
        # Расстояние в ATR
        df[f'{fib_name}_distance_atr'] = (
            (df['close'] - fib_price_from_low) / (atr + 1e-10)
        )
        
        # Бинарный признак: цена близко к уровню Fib? (в пределах 0.5 ATR)
        df[f'near_{fib_name}'] = (
            abs(df[f'{fib_name}_distance_atr']) < 0.5
        ).astype(int)
        
        # Близость к уровню Fib (экспоненциальный спад)
        df[f'proximity_{fib_name}'] = np.exp(-abs(df[f'{fib_name}_distance_atr']) / 2.0)
    
    # Процент отката от swing high (для нисходящего тренда)
    df['retracement_from_high_pct'] = (
        (df['swing_high'] - df['close']) / (df['swing_range'] + 1e-10) * 100
    )
    
    # Процент отката от swing low (для восходящего тренда)
    df['retracement_from_low_pct'] = (
        (df['close'] - df['swing_low']) / (df['swing_range'] + 1e-10) * 100
    )
    
    # На каком уровне Fib находится цена?
    df['current_fib_level'] = np.nan
    
    for i in range(len(df)):
        if pd.isna(df['swing_range'].iloc[i]):
            continue
        
        current_price = df['close'].iloc[i]
        swing_low = df['swing_low'].iloc[i]
        swing_range = df['swing_range'].iloc[i]
        
        if swing_range > 0:
            retracement_pct = (current_price - swing_low) / swing_range
            
            # Находим ближайший уровень Fib
            closest_fib = min(fib_levels, key=lambda x: abs(x - retracement_pct))
            df.iloc[i, df.columns.get_loc('current_fib_level')] = closest_fib
    
    # Расстояние до ближайшего уровня Fib
    df['distance_to_nearest_fib_atr'] = np.nan
    df['nearest_fib_level'] = np.nan
    
    for i in range(len(df)):
        if pd.isna(df['swing_range'].iloc[i]):
            continue
        
        current_price = df['close'].iloc[i]
        current_atr = atr.iloc[i]
        
        min_distance = np.inf
        nearest_level = np.nan
        
        for fib_level in fib_levels:
            fib_name = f'fib_{int(fib_level*1000)}'
            if f'{fib_name}_price' in df.columns:
                fib_price = df[f'{fib_name}_price'].iloc[i]
                if not pd.isna(fib_price):
                    distance = abs(current_price - fib_price) / (current_atr + 1e-10)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_level = fib_level
        
        if not np.isinf(min_distance):
            df.iloc[i, df.columns.get_loc('distance_to_nearest_fib_atr')] = min_distance
            df.iloc[i, df.columns.get_loc('nearest_fib_level')] = nearest_level
    
    return df


def add_pivot_points(df: pd.DataFrame, pivot_type: str = 'classic') -> pd.DataFrame:
    """
    Добавляет Pivot Points (Classic, Fibonacci, Camarilla)
    
    Args:
        df: DataFrame с колонками open, high, low, close
        pivot_type: Тип pivot points ('classic', 'fibonacci', 'camarilla')
    
    Returns:
        DataFrame с добавленными pivot point фичами
    """
    df = df.copy()
    
    # Предыдущий день (для классических pivot points нужен предыдущий день)
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    prev_close = df['close'].shift(1)
    
    if pivot_type == 'classic':
        # Classic Pivot Points
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        
        df['pivot'] = pivot
        df['pivot_r1'] = r1
        df['pivot_r2'] = r2
        df['pivot_r3'] = r3
        df['pivot_s1'] = s1
        df['pivot_s2'] = s2
        df['pivot_s3'] = s3
        
        # Расстояния до уровней
        if 'atr_14' in df.columns:
            atr = df['atr_14']
        else:
            atr = df['close'] * 0.001
        
        df['distance_to_pivot_atr'] = (df['close'] - pivot) / (atr + 1e-10)
        df['distance_to_r1_atr'] = (df['close'] - r1) / (atr + 1e-10)
        df['distance_to_s1_atr'] = (df['close'] - s1) / (atr + 1e-10)
        
    elif pivot_type == 'fibonacci':
        # Fibonacci Pivot Points
        pivot = (prev_high + prev_low + prev_close) / 3
        range_prev = prev_high - prev_low
        
        df['pivot_fib'] = pivot
        df['pivot_fib_r1'] = pivot + 0.382 * range_prev
        df['pivot_fib_r2'] = pivot + 0.618 * range_prev
        df['pivot_fib_r3'] = pivot + 1.0 * range_prev
        df['pivot_fib_s1'] = pivot - 0.382 * range_prev
        df['pivot_fib_s2'] = pivot - 0.618 * range_prev
        df['pivot_fib_s3'] = pivot - 1.0 * range_prev
        
    elif pivot_type == 'camarilla':
        # Camarilla Pivot Points
        range_prev = prev_high - prev_low
        df['pivot_cam'] = (prev_high + prev_low + prev_close) / 3
        df['pivot_cam_r1'] = prev_close + range_prev * 1.1 / 12
        df['pivot_cam_r2'] = prev_close + range_prev * 1.1 / 6
        df['pivot_cam_r3'] = prev_close + range_prev * 1.1 / 4
        df['pivot_cam_r4'] = prev_close + range_prev * 1.1 / 2
        df['pivot_cam_s1'] = prev_close - range_prev * 1.1 / 12
        df['pivot_cam_s2'] = prev_close - range_prev * 1.1 / 6
        df['pivot_cam_s3'] = prev_close - range_prev * 1.1 / 4
        df['pivot_cam_s4'] = prev_close - range_prev * 1.1 / 2
    
    return df

