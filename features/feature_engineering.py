"""
Главный модуль для генерации всех фичей
"""
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, time

from config.feature_config import FeatureConfig, default_config

def _get_timestamp() -> str:
    """Возвращает форматированную временную метку"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from features.price_features import add_price_features, add_lag_features
from features.technical_indicators import add_trend_indicators, add_oscillators
from features.volatility_features import add_volatility_features
from features.time_features import add_time_features
from features.pattern_features import add_candle_patterns
from features.multitimeframe_features import add_multitimeframe_features, resample_to_higher_timeframe
from features.statistical_features import add_statistical_features, add_momentum_features
from features.volume_features import add_volume_features
from features.tick_features import add_tick_features_to_minute_data

class FeatureEngineer:
    """
    Класс для генерации всех фичей из сырых данных
    """
    
    def __init__(self, config: FeatureConfig = None, cache_dir: str = 'data/features_cache'):
        """
        Args:
            config: Конфигурация параметров фичей
            cache_dir: Директория для кэширования промежуточных результатов
        """
        self.config = config if config is not None else default_config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_file_path(self, symbol: str, stage: str) -> Path:
        """Возвращает путь к файлу кэша для этапа генерации фичей"""
        return self.cache_dir / f'{symbol}_features_{stage}.pkl'
    
    def _save_intermediate_result(self, symbol: str, stage: str, df: pd.DataFrame):
        """Сохраняет промежуточный результат генерации фичей"""
        cache_file = self._get_cache_file_path(symbol, stage)
        try:
            df.to_pickle(cache_file)
        except Exception as e:
            print(f"[{_get_timestamp()}] ⚠ Ошибка при сохранении промежуточного результата ({stage}): {e}")
    
    def _load_intermediate_result(self, symbol: str, stage: str) -> Optional[pd.DataFrame]:
        """Загружает промежуточный результат генерации фичей"""
        cache_file = self._get_cache_file_path(symbol, stage)
        if cache_file.exists():
            try:
                df = pd.read_pickle(cache_file)
                print(f"[{_get_timestamp()}] ✓ Загружен промежуточный результат: {stage} ({len(df)} строк, {len(df.columns)} колонок)")
                return df
            except Exception as e:
                print(f"[{_get_timestamp()}] ⚠ Ошибка при загрузке промежуточного результата ({stage}): {e}")
        return None
    
    def create_features(self, df: pd.DataFrame, 
                       higher_timeframes_data: Optional[Dict[str, pd.DataFrame]] = None,
                       ticks_data: Optional[Dict[datetime, pd.DataFrame]] = None,
                       add_targets: bool = True,
                       symbol: str = 'XAUUSD',
                       save_intermediate: bool = True,
                       resume: bool = True) -> pd.DataFrame:
        """
        Создает все фичи из сырых данных
        
        Args:
            df: DataFrame с колонками open, high, low, close, volume (опционально)
            higher_timeframes_data: Словарь {timeframe: DataFrame} с данными старших таймфреймов
            ticks_data: Словарь {minute_time: ticks_df} с тиковыми данными для каждой минутной свечи
            add_targets: Добавлять ли целевые переменные
            symbol: Символ для кэширования (по умолчанию 'XAUUSD')
            save_intermediate: Сохранять ли промежуточные результаты
            resume: Продолжить с сохраненного прогресса, если он есть
        
        Returns:
            DataFrame со всеми фичами
        """
        timestamp = _get_timestamp()
        print(f"[{timestamp}] Генерация фичей...")
        
        # Проверяем, есть ли сохраненный прогресс
        stage_to_continue = None
        if resume and save_intermediate:
            # Пытаемся загрузить последний сохраненный этап (от самого позднего к раннему)
            stages = [
                ('after_tick_features', 'tick_features'),  # Самый последний и сложный этап
                ('after_multitimeframe', 'multitimeframe'),
                ('after_lag_features', 'lag_features'),
                ('after_statistical', 'statistical'),
                ('after_volatility', 'volatility'),
                ('after_oscillators', 'oscillators'),
                ('after_trend', 'trend'),
                ('after_price', 'price')
            ]
            
            loaded_stage = None
            for cache_stage, next_stage in stages:
                cached_df = self._load_intermediate_result(symbol, cache_stage)
                if cached_df is not None:
                    # Проверяем, что индексы совпадают
                    if len(cached_df) == len(df) and cached_df.index.equals(df.index):
                        df = cached_df
                        loaded_stage = cache_stage
                        stage_to_continue = next_stage
                        print(f"[{_get_timestamp()}] ✓ Продолжаем с этапа: {next_stage}")
                        break
            
            if loaded_stage == 'after_tick_features':
                # Все фичи уже сгенерированы
                print(f"[{_get_timestamp()}] ✓ Все фичи уже сгенерированы, пропускаем генерацию")
                stage_to_continue = 'done'
        
        # 1. Базовые ценовые фичи
        if stage_to_continue is None or stage_to_continue == 'price':
            print(f"[{_get_timestamp()}]   - Базовые ценовые фичи...")
            df = add_price_features(df)
            if save_intermediate:
                self._save_intermediate_result(symbol, 'after_price', df)
        
        # 2. Технические индикаторы - тренд
        if stage_to_continue is None or stage_to_continue == 'trend':
            print(f"[{_get_timestamp()}]   - Индикаторы тренда...")
            df = add_trend_indicators(
                df,
                sma_periods=self.config.sma_periods,
                ema_periods=self.config.ema_periods,
                macd_fast=self.config.macd_fast,
                macd_slow=self.config.macd_slow,
                macd_signal=self.config.macd_signal,
                adx_period=self.config.adx_period
            )
            if save_intermediate:
                self._save_intermediate_result(symbol, 'after_trend', df)
        
        # 3. Осцилляторы
        if stage_to_continue is None or stage_to_continue == 'oscillators':
            print(f"[{_get_timestamp()}]   - Осцилляторы...")
            df = add_oscillators(
                df,
                rsi_periods=self.config.rsi_periods,
                stoch_k=self.config.stoch_k,
                stoch_d=self.config.stoch_d,
                stoch_smooth=self.config.stoch_smooth,
                cci_period=self.config.cci_period,
                williams_period=self.config.williams_period,
                momentum_period=self.config.momentum_period,
                roc_period=self.config.roc_period
            )
            if save_intermediate:
                self._save_intermediate_result(symbol, 'after_oscillators', df)
        
        # 4. Волатильность
        if stage_to_continue is None or stage_to_continue == 'volatility':
            print(f"[{_get_timestamp()}]   - Фичи волатильности...")
            df = add_volatility_features(
                df,
                atr_periods=self.config.atr_periods,
                bb_period=self.config.bb_period,
                bb_std=self.config.bb_std,
                volatility_windows=self.config.rolling_windows
            )
            if save_intermediate:
                self._save_intermediate_result(symbol, 'after_volatility', df)
        
        # 5. Временные фичи
        print(f"[{_get_timestamp()}]   - Временные фичи...")
        df = add_time_features(df)
        
        # 6. Свечные паттерны
        print(f"[{_get_timestamp()}]   - Свечные паттерны...")
        df = add_candle_patterns(df)
        
        # 7. Объемные фичи (если доступны)
        if 'volume' in df.columns:
            print(f"[{_get_timestamp()}]   - Объемные фичи...")
            df = add_volume_features(df)
        
        # 8. Статистические фичи
        if stage_to_continue is None or stage_to_continue == 'statistical':
            print(f"[{_get_timestamp()}]   - Статистические фичи...")
            df = add_statistical_features(
                df,
                columns=['close', 'returns'],
                rolling_windows=self.config.rolling_windows
            )
            if save_intermediate:
                self._save_intermediate_result(symbol, 'after_statistical', df)
        
        # 9. Фичи момента
        print(f"[{_get_timestamp()}]   - Фичи момента...")
        df = add_momentum_features(
            df,
            columns=['close'],
            periods=[5, 10, 20, 50]
        )
        
        # 10. Lag features
        if stage_to_continue is None or stage_to_continue == 'lag_features':
            print(f"[{_get_timestamp()}]   - Lag features...")
            df = add_lag_features(
                df,
                columns=['close', 'returns', 'volume'] if 'volume' in df.columns else ['close', 'returns'],
                periods=self.config.lag_periods
            )
            if save_intermediate:
                self._save_intermediate_result(symbol, 'after_lag_features', df)
        
        # 11. Мультитаймфреймовые фичи
        if stage_to_continue is None or stage_to_continue == 'multitimeframe':
            if higher_timeframes_data is not None:
                print(f"[{_get_timestamp()}]   - Мультитаймфреймовые фичи...")
                df = add_multitimeframe_features(
                    df,
                    higher_timeframes_data,
                    timeframes=self.config.higher_timeframes
                )
                if save_intermediate:
                    self._save_intermediate_result(symbol, 'after_multitimeframe', df)
        
        # 12. Тиковые фичи (секундные свечи)
        if stage_to_continue is None or stage_to_continue == 'tick_features':
            if ticks_data is not None and len(ticks_data) > 0:
                print(f"[{_get_timestamp()}]   - Тиковые фичи (секундные свечи)...")
                df = add_tick_features_to_minute_data(
                    df,
                    ticks_data,
                    intervals=self.config.tick_candle_intervals
                )
                if save_intermediate:
                    self._save_intermediate_result(symbol, 'after_tick_features', df)
        
        # 13. Целевые переменные
        if add_targets:
            print(f"[{_get_timestamp()}]   - Целевые переменные...")
            df = self._add_target_variables(df)
        
        # 14. Удаление высококоррелированных фичей (опционально)
        if hasattr(self.config, 'remove_correlated_features') and self.config.remove_correlated_features:
            print(f"[{_get_timestamp()}]   - Удаление высококоррелированных фичей...")
            df = self._remove_correlated_features(df, threshold=getattr(self.config, 'correlation_threshold', 0.95))
        
        # Удаление строк с NaN (после расчета индикаторов)
        # Удаляем только строки, где критичные колонки (close, open, high, low) имеют NaN
        # Тиковые фичи уже заполнены нулями, поэтому они не будут причиной удаления
        initial_len = len(df)
        
        # Проверяем наличие критичных колонок
        critical_cols = ['close', 'open', 'high', 'low']
        critical_cols_present = [col for col in critical_cols if col in df.columns]
        
        if critical_cols_present:
            # Удаляем только строки, где критичные колонки имеют NaN
            df = df.dropna(subset=critical_cols_present)
        else:
            # Если критичных колонок нет, удаляем только полностью пустые строки
            df = df.dropna(how='all')
        
        timestamp = _get_timestamp()
        print(f"\n[{timestamp}] Удалено {initial_len - len(df)} строк с NaN из {initial_len} всего")
        
        print(f"[{timestamp}] Итого создано {len(df.columns)} фичей")
        print(f"[{timestamp}] Размер данных: {len(df)} строк")
        
        # Очищаем промежуточные файлы кэша после успешного завершения
        if save_intermediate:
            self._clear_intermediate_cache(symbol)
        
        return df
    
    def _clear_intermediate_cache(self, symbol: str):
        """Очищает промежуточные файлы кэша после успешного завершения"""
        stages = [
            'after_price', 'after_trend', 'after_oscillators', 'after_volatility',
            'after_statistical', 'after_lag_features', 'after_multitimeframe', 'after_tick_features'
        ]
        
        cleared = 0
        for stage in stages:
            cache_file = self._get_cache_file_path(symbol, stage)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                    cleared += 1
                except Exception as e:
                    print(f"[{_get_timestamp()}] ⚠ Ошибка при удалении кэша ({stage}): {e}")
        
        if cleared > 0:
            print(f"[{_get_timestamp()}] ✓ Очищено {cleared} промежуточных файлов кэша")
    
    def _remove_correlated_features(self, df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """
        Удаляет высококоррелированные фичи
        
        Args:
            df: DataFrame с фичами
            threshold: Порог корреляции для удаления (по умолчанию 0.95)
        
        Returns:
            DataFrame с удаленными коррелированными фичами
        """
        
        # Выбираем только числовые фичи (исключаем целевые переменные)
        exclude_patterns = ['future_return', 'signal_class', 'signal_class_name', 'max_future_return']
        feature_columns = [
            col for col in df.columns 
            if not any(pattern in col for pattern in exclude_patterns)
            and df[col].dtype in [np.number, 'float64', 'int64']
        ]
        
        if len(feature_columns) < 2:
            return df
        
        # Заполняем NaN для корректного расчета корреляции
        df_clean = df[feature_columns].fillna(df[feature_columns].median()).fillna(0.0)
        
        # Удаляем столбцы с нулевым стандартным отклонением (они не несут информации)
        # и могут вызвать деление на ноль при вычислении корреляции
        std_values = df_clean.std()
        zero_std_cols = std_values[std_values == 0].index.tolist()
        if zero_std_cols:
            print(f"[{_get_timestamp()}]   Предупреждение: {len(zero_std_cols)} фичей с нулевым std будут исключены из анализа корреляции")
            df_clean = df_clean.drop(columns=zero_std_cols)
            feature_columns = [col for col in feature_columns if col not in zero_std_cols]
        
        # Проверяем, что после удаления столбцов осталось достаточно фичей для корреляции
        if len(feature_columns) < 2:
            return df
        
        # Вычисляем корреляционную матрицу
        # Подавляем предупреждения для столбцов с нулевым std (они уже удалены выше)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            corr_matrix = df_clean.corr(numeric_only=True)
        
        # Находим высококоррелированные пары
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > threshold:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))
        
        if len(high_corr_pairs) == 0:
            print(f"[{_get_timestamp()}]   ✓ Высококоррелированных пар (>{threshold}) не найдено")
            return df
        
        # Выбираем фичи для удаления
        features_to_remove = self._select_features_to_remove(high_corr_pairs, feature_columns)
        
        if len(features_to_remove) > 0:
            print(f"[{_get_timestamp()}]   Найдено {len(high_corr_pairs)} высококоррелированных пар")
            print(f"[{_get_timestamp()}]   Удалено {len(features_to_remove)} фичей")
            df = df.drop(columns=list(features_to_remove))
        
        return df
    
    def _select_features_to_remove(self, high_corr_pairs: list, feature_columns: list) -> Set[str]:
        """
        Выбирает фичи для удаления из высококоррелированных пар
        
        Стратегия: удаляем более сложные или производные фичи,
        оставляя более простые и базовые
        """
        
        features_to_remove = set()
        
        # Приоритеты: оставляем более простые фичи
        priority_keywords = {
            'high': ['close', 'open', 'high', 'low', 'returns', 'log_returns'],
            'medium': ['sma', 'ema', 'rsi', 'macd', 'atr'],
            'low': ['lag', 'stat', 'tick', 'multitimeframe', 'position', 'shadow']
        }
        
        def get_priority(feature_name: str) -> int:
            """Возвращает приоритет фичи (меньше = выше приоритет)"""
            feature_lower = feature_name.lower()
            for priority, keywords in priority_keywords.items():
                if any(keyword in feature_lower for keyword in keywords):
                    if priority == 'high':
                        return 1
                    elif priority == 'medium':
                        return 2
                    else:
                        return 3
            return 2  # По умолчанию средний приоритет
        
        for feat1, feat2, corr in high_corr_pairs:
            # Если одна из фичей уже помечена к удалению, пропускаем
            if feat1 in features_to_remove or feat2 in features_to_remove:
                continue
            
            # Выбираем фичу с более низким приоритетом для удаления
            priority1 = get_priority(feat1)
            priority2 = get_priority(feat2)
            
            if priority1 > priority2:
                features_to_remove.add(feat1)
            elif priority2 > priority1:
                features_to_remove.add(feat2)
            else:
                # Если приоритеты равны, удаляем более длинное имя (обычно более сложное)
                if len(feat1) > len(feat2):
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
        
        return features_to_remove
    
    def _add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавляет целевые переменные для обучения
        
        Args:
            df: DataFrame с фичами
        
        Returns:
            DataFrame с добавленными целевыми переменными
        """
        df = df.copy()
        
        # Future returns (доходность через N периодов)
        for period in self.config.future_return_periods:
            df[f'future_return_{period}'] = df['close'].shift(-period) / df['close'] - 1
        
        # Direction (направление движения)
        for period in [1, 5, 10, 20]:
            future_price = df['close'].shift(-period)
            df[f'direction_{period}'] = np.where(
                future_price > df['close'], 1,
                np.where(future_price < df['close'], -1, 0)
            )
        
        # Volatility target (волатильность в будущем)
        for period in [5, 10, 20]:
            future_returns = df['close'].pct_change(period).shift(-period)
            df[f'future_volatility_{period}'] = future_returns.rolling(window=period).std()
        
        return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 60,
                        target_column: str = 'direction_1',
                        feature_columns: Optional[List[str]] = None) -> tuple:
        """
        Создает последовательности для Transformer модели
        
        Args:
            df: DataFrame с фичами
            sequence_length: Длина последовательности
            target_column: Колонка с целевой переменной
            feature_columns: Список колонок для использования (если None, все кроме целевых)
        
        Returns:
            Tuple (X, y) где X - массив последовательностей, y - массив целевых значений
        """
        if feature_columns is None:
            # Исключаем целевые переменные и временные метки
            exclude_patterns = ['target', 'label', 'direction', 'future_return', 'future_volatility']
            feature_columns = [
                col for col in df.columns 
                if not any(pattern in col.lower() for pattern in exclude_patterns)
                and col != target_column
            ]
        
        # Удаляем строки с NaN в целевой переменной
        df_clean = df.dropna(subset=[target_column])
        
        X = []
        y = []
        
        for i in range(sequence_length, len(df_clean)):
            # Последовательность фичей
            sequence = df_clean[feature_columns].iloc[i-sequence_length:i].values
            X.append(sequence)
            
            # Целевое значение
            y.append(df_clean[target_column].iloc[i])
        
        return np.array(X), np.array(y)
    
    def get_feature_list(self, df: pd.DataFrame) -> List[str]:
        """
        Возвращает список всех фичей (исключая целевые переменные)
        
        Args:
            df: DataFrame с фичами
        
        Returns:
            Список названий фичей
        """
        exclude_patterns = ['target', 'label', 'direction', 'future_return', 'future_volatility']
        feature_columns = [
            col for col in df.columns 
            if not any(pattern in col.lower() for pattern in exclude_patterns)
        ]
        return feature_columns

