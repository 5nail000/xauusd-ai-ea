"""
Модуль для генерации фичей из тиковых данных и секундных свечей
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta
from pathlib import Path

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
                                     minute_time: datetime,
                                     excluded_features: Optional[Set[str]] = None) -> pd.Series:
    """
    Агрегирует фичи из секундных свечей для минутной свечи
    
    Args:
        second_candles: Словарь {interval: DataFrame} с секундными свечами
        minute_time: Время минутной свечи
        excluded_features: Множество фичей для исключения (если None, проверка не выполняется)
    
    Returns:
        Series с агрегированными фичами
    """
    features = {}
    
    # Если список исключений не передан, создаем пустое множество
    if excluded_features is None:
        excluded_features = set()
    
    for interval, candles in second_candles.items():
        if candles.empty:
            continue
        
        prefix = f'tick_{interval}s'
        
        # Базовые статистики по позиционированию
        if 'close_position_in_range' in candles.columns:
            feature_name = f'{prefix}_close_pos_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['close_position_in_range'].mean()
            
            feature_name = f'{prefix}_close_pos_std'
            if feature_name not in excluded_features:
                features[feature_name] = candles['close_position_in_range'].std()
            
            feature_name = f'{prefix}_close_pos_min'
            if feature_name not in excluded_features:
                features[feature_name] = candles['close_position_in_range'].min()
            
            feature_name = f'{prefix}_close_pos_max'
            if feature_name not in excluded_features:
                features[feature_name] = candles['close_position_in_range'].max()
            
            feature_name = f'{prefix}_close_pos_median'
            if feature_name not in excluded_features:
                features[feature_name] = candles['close_position_in_range'].median()
        
        # Статистики по расстояниям
        if 'distance_to_high_pct' in candles.columns:
            feature_name = f'{prefix}_dist_high_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['distance_to_high_pct'].mean()
            
            feature_name = f'{prefix}_dist_low_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['distance_to_low_pct'].mean()
        
        # Статистики по спреду
        if 'spread' in candles.columns:
            feature_name = f'{prefix}_spread_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['spread'].mean()
            
            feature_name = f'{prefix}_spread_max'
            if feature_name not in excluded_features:
                features[feature_name] = candles['spread'].max()
            
            feature_name = f'{prefix}_spread_min'
            if feature_name not in excluded_features:
                features[feature_name] = candles['spread'].min()
            
            feature_name = f'{prefix}_spread_std'
            if feature_name not in excluded_features:
                features[feature_name] = candles['spread'].std()
        
        # Статистики по тиковому объему
        if 'tick_count' in candles.columns:
            feature_name = f'{prefix}_tick_count_sum'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_count'].sum()
            
            feature_name = f'{prefix}_tick_count_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_count'].mean()
            
            feature_name = f'{prefix}_tick_count_max'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_count'].max()
        
        # Статистики по скорости тиков
        if 'tick_rate' in candles.columns:
            feature_name = f'{prefix}_tick_rate_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_rate'].mean()
            
            feature_name = f'{prefix}_tick_rate_max'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_rate'].max()
        
        # Статистики по волатильности тиков
        if 'tick_volatility' in candles.columns:
            feature_name = f'{prefix}_tick_vol_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_volatility'].mean()
            
            feature_name = f'{prefix}_tick_vol_max'
            if feature_name not in excluded_features:
                features[feature_name] = candles['tick_volatility'].max()
        
        # Статистики по соотношению bid up/down
        if 'bid_up_down_ratio' in candles.columns:
            feature_name = f'{prefix}_bid_ratio_mean'
            if feature_name not in excluded_features:
                features[feature_name] = candles['bid_up_down_ratio'].mean()
            
            feature_name = f'{prefix}_bid_ratio_max'
            if feature_name not in excluded_features:
                features[feature_name] = candles['bid_up_down_ratio'].max()
    
    return pd.Series(features)

def add_tick_features_to_minute_data(df_minute: pd.DataFrame,
                                    ticks_data: Dict[datetime, pd.DataFrame],
                                    intervals: List[int],
                                    skip_if_exists: bool = True,
                                    apply_exclusions: bool = False,
                                    excluded_features: Optional[List[str]] = None,
                                    exclusions_file: Optional[Path] = None) -> pd.DataFrame:
    """
    Добавляет тиковые фичи к минутным данным
    
    Args:
        df_minute: DataFrame с минутными данными
        ticks_data: Словарь {minute_time: ticks_df} с тиковыми данными
        intervals: Список интервалов для секундных свечей
        skip_if_exists: Пропустить обработку, если тиковые фичи уже есть (по умолчанию: True)
        apply_exclusions: Применять ли список исключений при генерации тиковых фичей (по умолчанию: False)
        excluded_features: Список фичей для исключения (если None и apply_exclusions=True, загружается из файла)
        exclusions_file: Путь к файлу со списком исключений (по умолчанию: workspace/excluded_features.txt)
    
    Returns:
        DataFrame с добавленными тиковыми фичами
    """
    # Загружаем список исключений только если включена опция
    excluded_features_set = set()
    if apply_exclusions:
        if excluded_features is None:
            try:
                from utils.feature_exclusions import load_excluded_features
                if exclusions_file is None:
                    exclusions_file = Path('workspace/excluded_features.txt')
                excluded_features = load_excluded_features(exclusions_file)
                if excluded_features:
                    print(f"[{_get_timestamp()}] Загружено {len(excluded_features)} фичей для исключения из генерации тиковых фичей")
            except Exception as e:
                print(f"[{_get_timestamp()}] ⚠️  Предупреждение: Не удалось загрузить список исключений: {e}")
                excluded_features = []
        
        # Преобразуем в множество для быстрой проверки
        excluded_features_set = set(excluded_features) if excluded_features else set()
    
    df = df_minute.copy()
    
    # Проверяем, есть ли уже тиковые фичи
    if skip_if_exists:
        tick_cols = [col for col in df.columns if col.startswith('tick_')]
        if len(tick_cols) > 0:
            print(f"[{_get_timestamp()}] ✓ Тиковые фичи уже присутствуют ({len(tick_cols)} колонок), пропускаем обработку...")
            return df
    
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
    
    # Прогресс-бар: выводим каждые 1% или каждые 1000 итераций (что больше)
    progress_interval = max(1000, total_minutes // 100)
    start_time = datetime.now()
    last_progress_time = start_time
    
    print(f"[{_get_timestamp()}] Начало обработки тиковых фичей для {total_minutes:,} минутных свечей...")
    
    for idx, minute_time in enumerate(df.index, 1):
        # Выводим прогресс
        if idx % progress_interval == 0 or idx == total_minutes:
            current_time = datetime.now()
            elapsed = (current_time - start_time).total_seconds()
            progress_pct = (idx / total_minutes) * 100
            
            # Оценка оставшегося времени
            if idx > 0:
                avg_time_per_item = elapsed / idx
                remaining_items = total_minutes - idx
                estimated_remaining = avg_time_per_item * remaining_items
                remaining_str = f"{int(estimated_remaining // 60)}м {int(estimated_remaining % 60)}с"
            else:
                remaining_str = "расчет..."
            
            # Время с последнего вывода (для оценки скорости)
            time_since_last = (current_time - last_progress_time).total_seconds()
            speed = progress_interval / time_since_last if time_since_last > 0 else 0
            
            print(f"[{_get_timestamp()}] Прогресс: {idx:,}/{total_minutes:,} ({progress_pct:.1f}%) | "
                  f"Осталось: ~{remaining_str} | Скорость: {speed:.1f} свечей/сек")
            last_progress_time = current_time
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
        
        # Агрегируем фичи (передаем список исключений только если включена опция)
        features = aggregate_second_candles_features(
            second_candles, 
            minute_time, 
            excluded_features_set if apply_exclusions else None
        )
        tick_features_list.append(features)
    
    # Финальное сообщение о завершении обработки
    total_time = (datetime.now() - start_time).total_seconds()
    total_time_str = f"{int(total_time // 60)}м {int(total_time % 60)}с"
    print(f"[{_get_timestamp()}] ✓ Обработка тиковых фичей завершена за {total_time_str}")
    
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

