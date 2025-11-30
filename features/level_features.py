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
    
    Улучшенная версия с DBSCAN-подобным алгоритмом:
    - Находит все взаимно близкие точки, а не только последовательные
    - Более точная группировка близких экстремумов
    
    Args:
        extrema_values: Массив значений экстремумов
        atr: ATR для определения "близости"
        min_touches: Минимальное количество касаний для кластера
    
    Returns:
        Список кластеров (каждый кластер - список значений)
    """
    if len(extrema_values) < min_touches:
        return []
    
    # Сортируем значения для более эффективной обработки
    sorted_values = np.sort(extrema_values)
    eps = atr * 2.0  # Расстояние для определения близости
    
    clusters = []
    visited = set()
    
    # DBSCAN-подобный алгоритм: для каждой точки находим все близкие
    for i, val in enumerate(sorted_values):
        if i in visited:
            continue
        
        # Начинаем новый кластер с текущей точки
        cluster = [val]
        visited.add(i)
        
        # Находим все точки в пределах eps от текущей
        # Используем бинарный поиск для эффективности
        for j in range(i + 1, len(sorted_values)):
            if j in visited:
                continue
            
            other_val = sorted_values[j]
            # Если точка слишком далеко, остальные тоже будут далеко (отсортированы)
            if other_val - val > eps:
                break
            
            # Проверяем, близка ли точка к любой точке в кластере
            # (для более точной кластеризации)
            is_close = False
            for cluster_val in cluster:
                if abs(other_val - cluster_val) <= eps:
                    is_close = True
                    break
            
            if is_close:
                cluster.append(other_val)
                visited.add(j)
        
        # Сохраняем кластер, если в нем достаточно точек
        if len(cluster) >= min_touches:
            clusters.append(cluster)
    
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
    
    # Вспомогательная функция для выбора лучшего уровня
    def select_best_level(clusters: List[List[float]], current_price: float, 
                         min_strength: int = 2) -> Tuple[Optional[List[float]], Optional[float]]:
        """
        Выбирает лучший уровень: ближайший к цене среди достаточно сильных
        
        Args:
            clusters: Список кластеров
            current_price: Текущая цена
            min_strength: Минимальная сила уровня
        
        Returns:
            Tuple (лучший кластер, центр уровня) или (None, None)
        """
        if not clusters:
            return None, None
        
        # Фильтруем кластеры по минимальной силе
        valid_clusters = [c for c in clusters if len(c) >= min_strength]
        if not valid_clusters:
            return None, None
        
        # Вычисляем score для каждого кластера: близость важнее, но сила тоже учитывается
        scored_clusters = []
        for cluster in valid_clusters:
            center = np.median(cluster)
            distance = abs(current_price - center)
            strength = len(cluster)
            # Комбинированный score: расстояние / (сила + 1)
            # Чем ближе и сильнее, тем лучше (меньше score)
            score = distance / (strength + 1.0)
            scored_clusters.append((score, cluster, center))
        
        # Выбираем лучший (с наименьшим score)
        scored_clusters.sort(key=lambda x: x[0])
        return scored_clusters[0][1], scored_clusters[0][2]
    
    # Вспомогательная функция для поиска уровня с fallback
    def find_level_with_fallback(window_extrema: pd.Series, current_price: float,
                                 avg_atr: float, min_touches: int,
                                 cluster_atr_multiplier: float) -> Tuple[Optional[List[float]], Optional[float]]:
        """
        Находит уровень с использованием fallback-механизма
        
        Пробует разные параметры, начиная со строгих и ослабляя требования
        """
        if len(window_extrema) < 2:  # Минимум 2 для fallback
            return None, None
        
        # Попытка 1: Строгие параметры (min_touches)
        clusters = cluster_extrema(
            window_extrema.values,
            avg_atr * cluster_atr_multiplier,
            min_touches
        )
        if clusters:
            cluster, center = select_best_level(clusters, current_price, min_strength=min_touches)
            if cluster is not None:
                return cluster, center
        
        # Fallback 1: Уменьшаем min_touches до 2
        if min_touches > 2:
            clusters = cluster_extrema(
                window_extrema.values,
                avg_atr * cluster_atr_multiplier,
                2
            )
            if clusters:
                cluster, center = select_best_level(clusters, current_price, min_strength=2)
                if cluster is not None:
                    return cluster, center
        
        # Fallback 2: Увеличиваем tolerance (более широкая кластеризация)
        clusters = cluster_extrema(
            window_extrema.values,
            avg_atr * cluster_atr_multiplier * 1.5,  # Увеличиваем на 50%
            2
        )
        if clusters:
            cluster, center = select_best_level(clusters, current_price, min_strength=2)
            if cluster is not None:
                return cluster, center
        
        # Fallback 3: Используем просто ближайший экстремум (если есть)
        if len(window_extrema) > 0:
            nearest = window_extrema.iloc[-1]  # Последний экстремум (самый свежий)
            return [nearest], nearest
        
        return None, None
    
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
        
        # Используем средний ATR за период вместо текущего (более стабильно)
        atr_window_start = max(0, i - 20)  # Средний ATR за последние 20 периодов
        avg_atr = atr.iloc[atr_window_start:i+1].mean()
        current_price = df['close'].iloc[i]
        
        # Обработка поддержек (минимумы)
        cluster, center = find_level_with_fallback(
            window_mins, current_price, avg_atr, min_touches, cluster_atr_multiplier
        )
        
        if cluster is not None and center is not None:
            support_center = center
            support_width = np.std(cluster) if len(cluster) > 1 else avg_atr * 0.5
            support_strength = len(cluster)
            
            support_levels.append(support_center)
            support_widths.append(support_width)
            support_strengths.append(support_strength)
        else:
            # Fallback: используем скользящий минимум за последние 20 свечей
            # Это лучше, чем полный пропуск
            if len(window_mins) > 0:
                rolling_min = window_mins.min()
                if pd.notna(rolling_min):
                    support_levels.append(rolling_min)
                    support_widths.append(avg_atr * 0.5)
                    support_strengths.append(1)  # Минимальная сила
                else:
                    support_levels.append(np.nan)
                    support_widths.append(np.nan)
                    support_strengths.append(0)
            else:
                support_levels.append(np.nan)
                support_widths.append(np.nan)
                support_strengths.append(0)
        
        # Обработка сопротивлений (максимумы)
        cluster, center = find_level_with_fallback(
            window_maxs, current_price, avg_atr, min_touches, cluster_atr_multiplier
        )
        
        if cluster is not None and center is not None:
            resistance_center = center
            resistance_width = np.std(cluster) if len(cluster) > 1 else avg_atr * 0.5
            resistance_strength = len(cluster)
            
            resistance_levels.append(resistance_center)
            resistance_widths.append(resistance_width)
            resistance_strengths.append(resistance_strength)
        else:
            # Fallback: используем скользящий максимум за последние 20 свечей
            # Это лучше, чем полный пропуск
            if len(window_maxs) > 0:
                rolling_max = window_maxs.max()
                if pd.notna(rolling_max):
                    resistance_levels.append(rolling_max)
                    resistance_widths.append(avg_atr * 0.5)
                    resistance_strengths.append(1)  # Минимальная сила
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
    
    # 3.1. Forward fill уровней с ограничениями (уменьшаем пропуски)
    # Заполняем пропуски последним найденным уровнем, но только если:
    # - Прошло не более max_fill_candles свечей
    # - Цена не ушла далеко от уровня (не более max_deviation_atr * ATR)
    max_fill_candles = 50  # Максимум 50 свечей вперед
    max_deviation_atr = 3.0  # Максимальное отклонение в единицах ATR
    
    # Forward fill для support_level
    last_support_idx = -1
    last_support_level = None
    last_support_atr = None
    for i in range(len(df)):
        if pd.notna(df['support_level'].iloc[i]):
            last_support_idx = i
            last_support_level = df['support_level'].iloc[i]
            # Используем средний ATR за последние 20 периодов
            atr_window_start = max(0, i - 20)
            last_support_atr = atr.iloc[atr_window_start:i+1].mean()
        elif last_support_level is not None and (i - last_support_idx) <= max_fill_candles:
            # Проверяем, не ушла ли цена слишком далеко
            current_price = df['close'].iloc[i]
            current_atr = atr.iloc[max(0, i - 20):i+1].mean() if i >= 20 else last_support_atr
            price_deviation = abs(current_price - last_support_level)
            max_allowed_deviation = max_deviation_atr * (current_atr if pd.notna(current_atr) else last_support_atr)
            
            if price_deviation <= max_allowed_deviation:
                df.loc[df.index[i], 'support_level'] = last_support_level
                # Используем последнюю известную ширину или оценку
                if pd.notna(df['support_width'].iloc[last_support_idx]):
                    df.loc[df.index[i], 'support_width'] = df['support_width'].iloc[last_support_idx]
                else:
                    df.loc[df.index[i], 'support_width'] = current_atr * 0.5 if pd.notna(current_atr) else np.nan
                # Сохраняем последнюю известную силу (не уменьшаем, так как это количество касаний)
                df.loc[df.index[i], 'support_strength'] = df['support_strength'].iloc[last_support_idx]
    
    # Forward fill для resistance_level
    last_resistance_idx = -1
    last_resistance_level = None
    last_resistance_atr = None
    for i in range(len(df)):
        if pd.notna(df['resistance_level'].iloc[i]):
            last_resistance_idx = i
            last_resistance_level = df['resistance_level'].iloc[i]
            # Используем средний ATR за последние 20 периодов
            atr_window_start = max(0, i - 20)
            last_resistance_atr = atr.iloc[atr_window_start:i+1].mean()
        elif last_resistance_level is not None and (i - last_resistance_idx) <= max_fill_candles:
            # Проверяем, не ушла ли цена слишком далеко
            current_price = df['close'].iloc[i]
            current_atr = atr.iloc[max(0, i - 20):i+1].mean() if i >= 20 else last_resistance_atr
            price_deviation = abs(current_price - last_resistance_level)
            max_allowed_deviation = max_deviation_atr * (current_atr if pd.notna(current_atr) else last_resistance_atr)
            
            if price_deviation <= max_allowed_deviation:
                df.loc[df.index[i], 'resistance_level'] = last_resistance_level
                # Используем последнюю известную ширину или оценку
                if pd.notna(df['resistance_width'].iloc[last_resistance_idx]):
                    df.loc[df.index[i], 'resistance_width'] = df['resistance_width'].iloc[last_resistance_idx]
                else:
                    df.loc[df.index[i], 'resistance_width'] = current_atr * 0.5 if pd.notna(current_atr) else np.nan
                # Сохраняем последнюю известную силу (не уменьшаем, так как это количество касаний)
                df.loc[df.index[i], 'resistance_strength'] = df['resistance_strength'].iloc[last_resistance_idx]
    
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
    # Используем NaN когда уровень не найден (а не 0), чтобы не вводить модель в заблуждение
    df['in_support_zone'] = np.where(
        df['support_level'].notna(),
        (abs(df['distance_to_support_sigma']) < 1.0).astype(int),
        np.nan
    )
    df['in_resistance_zone'] = np.where(
        df['resistance_level'].notna(),
        (abs(df['distance_to_resistance_sigma']) < 1.0).astype(int),
        np.nan
    )
    
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

