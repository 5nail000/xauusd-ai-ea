"""
Модуль для генерации фичей из тиковых данных и секундных свечей
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

def _get_timestamp() -> str:
    """Возвращает форматированную временную метку"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_second_candles_from_ticks(ticks_df: pd.DataFrame, 
                                     interval_seconds: int) -> pd.DataFrame:
    """
    Создает свечи из тиков для указанного интервала в секундах
    
    Args:
        ticks_df: DataFrame с тиками (колонки: bid, ask, volume, spread)
        interval_seconds: Интервал свечи в секундах (1, 2, 3, 5, 10, 15, 20, 30, 45)
    
    Returns:
        DataFrame с секундными свечами (open, high, low, close, volume, tick_count)
        High рассчитывается по bid, Low по ask
    """
    if ticks_df.empty:
        return pd.DataFrame()
    
    # Ресэмплинг по секундам
    rule = f'{interval_seconds}s'
    
    # Open - первая цена (bid для открытия)
    open_bid = ticks_df['bid'].resample(rule).first()
    open_ask = ticks_df['ask'].resample(rule).first()
    
    # High - максимальный bid
    high_bid = ticks_df['bid'].resample(rule).max()
    
    # Low - минимальный ask
    low_ask = ticks_df['ask'].resample(rule).min()
    
    # Close - последняя цена (bid для закрытия)
    close_bid = ticks_df['bid'].resample(rule).last()
    close_ask = ticks_df['ask'].resample(rule).last()
    
    # Объем и количество тиков
    volume = ticks_df['volume'].resample(rule).sum()
    tick_count = ticks_df['bid'].resample(rule).count()
    
    # Спред
    spread = ticks_df['spread'].resample(rule).mean()
    spread_max = ticks_df['spread'].resample(rule).max()
    spread_min = ticks_df['spread'].resample(rule).min()
    
    # Создание DataFrame
    candles = pd.DataFrame({
        'open_bid': open_bid,
        'open_ask': open_ask,
        'high_bid': high_bid,
        'low_ask': low_ask,
        'close_bid': close_bid,
        'close_ask': close_ask,
        'volume': volume,
        'tick_count': tick_count,
        'spread': spread,
        'spread_max': spread_max,
        'spread_min': spread_min
    })
    
    # Основные цены для свечи (используем bid для close)
    candles['open'] = (open_bid + open_ask) / 2  # Средняя цена для open
    candles['high'] = high_bid  # High по bid
    candles['low'] = low_ask    # Low по ask
    candles['close'] = close_bid  # Close по bid
    
    # Удаление строк с NaN
    candles = candles.dropna()
    
    return candles

def add_tick_positioning_features(candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет фичи позиционирования цены относительно High/Low
    
    Args:
        candles_df: DataFrame с секундными свечами (high, low, close)
    
    Returns:
        DataFrame с добавленными фичами позиционирования
    """
    df = candles_df.copy()
    
    # Позиция close в диапазоне High-Low (0 = Low, 1 = High)
    range_size = df['high'] - df['low']
    df['close_position_in_range'] = (
        (df['close'] - df['low']) / (range_size + 1e-10)
    )
    
    # Расстояние от close до high (в процентах)
    df['distance_to_high_pct'] = (
        (df['high'] - df['close']) / (df['high'] + 1e-10) * 100
    )
    
    # Расстояние от close до low (в процентах)
    df['distance_to_low_pct'] = (
        (df['close'] - df['low']) / (df['low'] + 1e-10) * 100
    )
    
    # Абсолютные расстояния
    df['distance_to_high'] = df['high'] - df['close']
    df['distance_to_low'] = df['close'] - df['low']
    
    # Нормализованные расстояния (относительно размера свечи)
    df['distance_to_high_norm'] = df['distance_to_high'] / (range_size + 1e-10)
    df['distance_to_low_norm'] = df['distance_to_low'] / (range_size + 1e-10)
    
    return df

def add_tick_statistics_features(ticks_df: pd.DataFrame, 
                                candles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет статистические фичи на основе тиков
    
    Args:
        ticks_df: DataFrame с тиками
        candles_df: DataFrame с секундными свечами
    
    Returns:
        DataFrame с добавленными статистическими фичами
    """
    df = candles_df.copy()
    
    # Ресэмплинг тиков под интервал свечей
    if not ticks_df.empty:
        # Определяем интервал свечей по разнице времен
        if len(df) > 1:
            interval = (df.index[1] - df.index[0]).total_seconds()
            rule = f'{int(interval)}s'
        else:
            interval = 1.0  # По умолчанию 1 секунда
            rule = '1s'
        
        # Скорость изменения цены (тики в секунду)
        tick_rate = ticks_df['bid'].resample(rule).count() / interval
        df['tick_rate'] = tick_rate.reindex(df.index, method='ffill')
        
        # Средняя скорость изменения bid
        bid_changes = ticks_df['bid'].diff().abs()
        bid_velocity = bid_changes.resample(rule).mean()
        df['bid_velocity'] = bid_velocity.reindex(df.index, method='ffill')
        
        # Средняя скорость изменения ask
        ask_changes = ticks_df['ask'].diff().abs()
        ask_velocity = ask_changes.resample(rule).mean()
        df['ask_velocity'] = ask_velocity.reindex(df.index, method='ffill')
        
        # Распределение тиков по bid/ask
        # Подсчет тиков с ростом bid
        bid_up = (ticks_df['bid'].diff() > 0).astype(int)
        bid_up_count = bid_up.resample(rule).sum()
        df['bid_up_ticks'] = bid_up_count.reindex(df.index, method='ffill')
        
        # Подсчет тиков с падением bid
        bid_down = (ticks_df['bid'].diff() < 0).astype(int)
        bid_down_count = bid_down.resample(rule).sum()
        df['bid_down_ticks'] = bid_down_count.reindex(df.index, method='ffill')
        
        # Соотношение роста/падения
        df['bid_up_down_ratio'] = (
            df['bid_up_ticks'] / (df['bid_down_ticks'] + 1e-10)
        )
        
        # Волатильность тиков (стандартное отклонение изменений bid)
        bid_volatility = bid_changes.resample(rule).std()
        df['tick_volatility'] = bid_volatility.reindex(df.index, method='ffill')
    
    return df

def aggregate_second_candles_features(second_candles: Dict[int, pd.DataFrame],
                                     minute_time: datetime) -> pd.Series:
    """
    Агрегирует фичи из секундных свечей для минутной свечи
    
    Args:
        second_candles: Словарь {interval: DataFrame} с секундными свечами
        minute_time: Время минутной свечи
    
    Returns:
        Series с агрегированными фичами
    """
    features = {}
    
    for interval, candles in second_candles.items():
        if candles.empty:
            continue
        
        prefix = f'tick_{interval}s'
        
        # Базовые статистики по позиционированию
        if 'close_position_in_range' in candles.columns:
            features[f'{prefix}_close_pos_mean'] = candles['close_position_in_range'].mean()
            features[f'{prefix}_close_pos_std'] = candles['close_position_in_range'].std()
            features[f'{prefix}_close_pos_min'] = candles['close_position_in_range'].min()
            features[f'{prefix}_close_pos_max'] = candles['close_position_in_range'].max()
            features[f'{prefix}_close_pos_median'] = candles['close_position_in_range'].median()
        
        # Статистики по расстояниям
        if 'distance_to_high_pct' in candles.columns:
            features[f'{prefix}_dist_high_mean'] = candles['distance_to_high_pct'].mean()
            features[f'{prefix}_dist_low_mean'] = candles['distance_to_low_pct'].mean()
        
        # Статистики по спреду
        if 'spread' in candles.columns:
            features[f'{prefix}_spread_mean'] = candles['spread'].mean()
            features[f'{prefix}_spread_max'] = candles['spread'].max()
            features[f'{prefix}_spread_min'] = candles['spread'].min()
            features[f'{prefix}_spread_std'] = candles['spread'].std()
        
        # Статистики по тиковому объему
        if 'tick_count' in candles.columns:
            features[f'{prefix}_tick_count_sum'] = candles['tick_count'].sum()
            features[f'{prefix}_tick_count_mean'] = candles['tick_count'].mean()
            features[f'{prefix}_tick_count_max'] = candles['tick_count'].max()
        
        # Статистики по скорости тиков
        if 'tick_rate' in candles.columns:
            features[f'{prefix}_tick_rate_mean'] = candles['tick_rate'].mean()
            features[f'{prefix}_tick_rate_max'] = candles['tick_rate'].max()
        
        # Статистики по волатильности тиков
        if 'tick_volatility' in candles.columns:
            features[f'{prefix}_tick_vol_mean'] = candles['tick_volatility'].mean()
            features[f'{prefix}_tick_vol_max'] = candles['tick_volatility'].max()
        
        # Статистики по соотношению bid up/down
        if 'bid_up_down_ratio' in candles.columns:
            features[f'{prefix}_bid_ratio_mean'] = candles['bid_up_down_ratio'].mean()
            features[f'{prefix}_bid_ratio_max'] = candles['bid_up_down_ratio'].max()
    
    return pd.Series(features)

def add_tick_features_to_minute_data(df_minute: pd.DataFrame,
                                    ticks_data: Dict[datetime, pd.DataFrame],
                                    intervals: List[int]) -> pd.DataFrame:
    """
    Добавляет тиковые фичи к минутным данным
    
    Args:
        df_minute: DataFrame с минутными данными
        ticks_data: Словарь {minute_time: ticks_df} с тиковыми данными
        intervals: Список интервалов для секундных свечей
    
    Returns:
        DataFrame с добавленными тиковыми фичами
    """
    df = df_minute.copy()
    
    # Диагностика: статистика по тикам
    total_minutes = len(df.index)
    minutes_with_ticks = len(ticks_data)
    print(f"[{_get_timestamp()}] Диагностика тиковых данных:")
    print(f"[{_get_timestamp()}]   Всего минутных свечей: {total_minutes}")
    print(f"[{_get_timestamp()}]   Минутных свечей с тиками в словаре: {minutes_with_ticks}")
    
    # Создаем список для хранения новых фичей
    tick_features_list = []
    
    # Счетчики для диагностики
    ticks_with_data = 0
    ticks_empty = 0
    ticks_missing = 0
    total_ticks_count = 0
    sample_ticks_info = []  # Для первых 5 примеров
    
    for minute_time in df.index:
        if minute_time not in ticks_data:
            # Если нет тиков для этой минуты, создаем пустой Series
            # Он будет заполнен нулями позже
            ticks_missing += 1
            tick_features_list.append(pd.Series(dtype=float))
            continue
        
        ticks_df = ticks_data[minute_time]
        
        if ticks_df.empty:
            # Если тики пустые, создаем пустой Series
            ticks_empty += 1
            tick_features_list.append(pd.Series(dtype=float))
            continue
        
        # Тики есть и не пустые
        ticks_with_data += 1
        tick_count = len(ticks_df)
        total_ticks_count += tick_count
        
        # Сохраняем информацию о первых 5 примерах для диагностики
        if len(sample_ticks_info) < 5:
            time_range = f"{ticks_df.index.min()} - {ticks_df.index.max()}" if not ticks_df.empty else "N/A"
            sample_ticks_info.append({
                'minute': minute_time,
                'tick_count': tick_count,
                'time_range': time_range,
                'columns': list(ticks_df.columns)
            })
        
        # Создаем секундные свечи для всех интервалов
        second_candles = {}
        for interval in intervals:
            candles = create_second_candles_from_ticks(ticks_df, interval)
            if not candles.empty:
                candles = add_tick_positioning_features(candles)
                candles = add_tick_statistics_features(ticks_df, candles)
                second_candles[interval] = candles
        
        # Агрегируем фичи
        features = aggregate_second_candles_features(second_candles, minute_time)
        tick_features_list.append(features)
    
    # Выводим детальную диагностику
    timestamp = _get_timestamp()
    print(f"[{timestamp}] Статистика обработки тиков:")
    print(f"[{timestamp}]   Минутных свечей с тиками (не пустые): {ticks_with_data}")
    print(f"[{timestamp}]   Минутных свечей с пустыми тиками: {ticks_empty}")
    print(f"[{timestamp}]   Минутных свечей без тиков в словаре: {ticks_missing}")
    if ticks_with_data > 0:
        avg_ticks = total_ticks_count / ticks_with_data
        print(f"[{timestamp}]   Среднее количество тиков на свечу: {avg_ticks:.1f}")
        print(f"[{timestamp}]   Всего тиков обработано: {total_ticks_count:,}")
    
    if sample_ticks_info:
        print(f"[{timestamp}] Примеры тиковых данных (первые {len(sample_ticks_info)}):")
        for i, info in enumerate(sample_ticks_info, 1):
            print(f"[{timestamp}]   {i}. Минута {info['minute']}: {info['tick_count']} тиков, диапазон: {info['time_range']}")
            if info['columns']:
                cols_str = ', '.join(info['columns'][:5])
                if len(info['columns']) > 5:
                    cols_str += '...'
                print(f"[{timestamp}]      Колонки: {cols_str}")
    
    # Объединяем все фичи в DataFrame
    if tick_features_list:
        tick_features_df = pd.DataFrame(tick_features_list, index=df.index)
        
        # Диагностика: проверяем, сколько фичей создано
        print(f"[{_get_timestamp()}] Создано тиковых фичей: {len(tick_features_df.columns)} колонок")
        
        # Заполняем NaN нулями для тиковых фичей (если тиков не было)
        # Это предотвратит удаление всех строк при dropna()
        nan_before = tick_features_df.isna().sum().sum()
        tick_features_df = tick_features_df.fillna(0)
        nan_after = tick_features_df.isna().sum().sum()
        print(f"[{_get_timestamp()}] Заполнено NaN в тиковых фичах: {nan_before} -> {nan_after}")
        
        # Объединяем с основным DataFrame
        df = pd.concat([df, tick_features_df], axis=1)
        print(f"[{_get_timestamp()}] Объединено с основным DataFrame. Итого колонок: {len(df.columns)}")
    else:
        print(f"[{_get_timestamp()}] ⚠ Предупреждение: тиковые фичи не созданы (tick_features_list пуст)")
    
    return df

