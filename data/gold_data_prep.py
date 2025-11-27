"""
Модуль для подготовки данных по золоту (XAUUSD)
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
from data.mt5_data_loader import MT5DataLoader
from data.tick_data_loader import TickDataLoader
from features.feature_engineering import FeatureEngineer
from data.target_generator import TargetGenerator
from config.feature_config import FeatureConfig

def _get_timestamp() -> str:
    """Возвращает форматированную временную метку"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class GoldDataPreparator:
    """
    Класс для подготовки данных по золоту для обучения модели
    """
    
    def __init__(self, 
                 config: Optional[FeatureConfig] = None,
                 training_months: int = 6,
                 cache_dir: str = 'workspace/raw_data/cache',
                 offline_mode: bool = False):
        """
        Args:
            config: Конфигурация фичей
            training_months: Количество месяцев данных для обучения (по умолчанию 6)
            cache_dir: Директория для сохранения подготовленных данных
            offline_mode: Режим offline - работа только с кэшированными данными без подключения к MT5
        """
        self.config = config if config is not None else FeatureConfig()
        self.training_months = training_months
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline_mode = offline_mode
        self.feature_engineer = FeatureEngineer(self.config, cache_dir=str(self.cache_dir))
        self.target_generator = TargetGenerator(
            breakout_threshold=450.0,
            bounce_threshold=350.0,
            lookahead_periods=60
        )
    
    def _get_cache_file_path(self, symbol: str, period_label: str, end_date: Optional[datetime] = None,
                             load_ticks: bool = True, load_higher_tf: bool = True) -> Path:
        """Генерирует путь к файлу кэша на основе параметров"""
        if end_date is None:
            end_date = datetime.now()
        
        # Создаем уникальное имя файла на основе параметров
        date_str = end_date.strftime('%Y%m%d')
        ticks_flag = 'ticks' if load_ticks else 'noticks'
        tf_flag = 'mtf' if load_higher_tf else 'notf'
        filename = f'{symbol}_{period_label}_{date_str}_{ticks_flag}_{tf_flag}.pkl'
        return self.cache_dir / filename
    
    def load_gold_data(self, 
                      symbol: str = 'XAUUSD',
                      end_date: Optional[datetime] = None,
                      months: Optional[int] = None,
                      days: Optional[int] = None,
                      use_ticks_fallback: bool = True) -> pd.DataFrame:
        """
        Загружает минутные данные по золоту.
        Если минутных данных недостаточно, создает их из тиков.
        
        Args:
            symbol: Символ (по умолчанию XAUUSD)
            end_date: Конечная дата (если None, используется текущая дата)
            months: Количество месяцев (если None и days не указан, используется self.training_months)
            days: Количество дней (приоритет над months)
            use_ticks_fallback: Использовать тики для создания минутных свечей, если данных недостаточно
        
        Returns:
            DataFrame с минутными данными
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Определяем период: приоритет у days, если указан
        if days is not None:
            period_days = days
        elif months is not None:
            period_days = months * 30
        else:
            months = self.training_months
            period_days = months * 30
        
        start_date = end_date - timedelta(days=period_days)
        
        # В offline режиме работаем только с кэшированными данными
        if self.offline_mode:
            print(f"   [OFFLINE MODE] Загрузка данных только из кэша...")
            print(f"   Период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}")
            
            # Загружаем тики из кэша
            from data.tick_data_loader import TickDataLoader
            tick_loader = TickDataLoader(use_cache=True, offline_mode=True)
            
            print(f"   Загрузка тиков из кэша...")
            ticks_df = tick_loader.load_ticks(
                symbol=symbol,
                start_time=start_date,
                end_time=end_date,
                use_cache=True
            )
            
            if ticks_df.empty:
                raise ValueError(
                    f"В offline режиме не найдены кэшированные тики для {symbol}\n"
                    f"   Период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}\n"
                    f"   Убедитесь, что тики загружены в workspace/raw_data/ticks/"
                )
            
            print(f"   Загружено {len(ticks_df):,} тиков из кэша")
            print(f"   Создание минутных свечей из тиков...")
            
            # Создаем минутные свечи из тиков
            df = MT5DataLoader.create_minute_candles_from_ticks(ticks_df)
            
            if df.empty:
                raise ValueError("Не удалось создать минутные свечи из тиков")
            
            # Фильтруем по нужному периоду
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if df.empty:
                raise ValueError(
                    f"После фильтрации по периоду данных не осталось\n"
                    f"   Доступный период в тиках: {ticks_df.index.min()} - {ticks_df.index.max()}\n"
                    f"   Требуемый период: {start_date} - {end_date}"
                )
            
            print(f"   ✓ Создано {len(df)} минутных свечей из тиков")
            print(f"   Период: {df.index.min().strftime('%Y-%m-%d %H:%M')} - {df.index.max().strftime('%Y-%m-%d %H:%M')}")
            return df
        
        # Обычный режим с подключением к MT5
        loader = MT5DataLoader()
        if not loader.connect():
            raise ConnectionError("Не удалось подключиться к MT5")
        
        tick_loader = None
        try:
            # Проверяем доступность символа
            available_symbols = loader.get_symbols()
            if available_symbols:
                # Ищем точное совпадение или похожие символы
                symbol_found = symbol in available_symbols
                if not symbol_found:
                    # Ищем похожие символы
                    similar = [s for s in available_symbols if 'GOLD' in s.upper() or 'XAU' in s.upper() or symbol.upper() in s.upper()]
                    if similar:
                        print(f"⚠️  Символ {symbol} не найден в списке доступных символов")
                        print(f"   Похожие символы: {', '.join(similar[:10])}")
                        print(f"   Попробуйте использовать один из этих символов с параметром --symbol")
                    else:
                        print(f"⚠️  Символ {symbol} не найден. Всего доступно символов: {len(available_symbols)}")
                        # Показываем первые 20 символов для справки
                        print(f"   Примеры доступных символов: {', '.join(available_symbols[:20])}")
            
            # Пытаемся загрузить минутные данные из MT5
            df = loader.load_data(
                symbol=symbol,
                timeframe='M1',
                start_date=start_date,
                end_date=end_date
            )
            
            # Проверяем, достаточно ли данных
            if not df.empty:
                actual_start = df.index.min()
                actual_end = df.index.max()
                
                # Если данных недостаточно (меньше 80% от требуемого периода)
                days_loaded = (actual_end - actual_start).days
                days_required = (end_date - start_date).days
                
                if days_loaded < days_required * 0.8 and use_ticks_fallback:
                    print(f"   ⚠️  Загружено только {days_loaded} дней из {days_required} требуемых")
                    print(f"   Доступный период: {actual_start.strftime('%Y-%m-%d %H:%M')} - {actual_end.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   Требуемый период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}")
                    print(f"   Создаем недостающие минутные свечи из тиков...")
                    
                    from data.tick_data_loader import TickDataLoader
                    # Создаем TickDataLoader с использованием существующего подключения MT5
                    tick_loader = TickDataLoader(mt5_connection=loader, use_cache=True, offline_mode=self.offline_mode)
                    
                    candles_to_add = []
                    
                    # Загружаем тики за период ДО доступных данных
                    if start_date < actual_start:
                        missing_start = start_date
                        missing_end = actual_start - timedelta(minutes=1)
                        
                        if missing_start < missing_end:
                            print(f"   Загрузка тиков за период ДО доступных данных: {missing_start.strftime('%Y-%m-%d %H:%M')} - {missing_end.strftime('%Y-%m-%d %H:%M')}...")
                            ticks_df = tick_loader.load_ticks(
                                symbol=symbol,
                                start_time=missing_start,
                                end_time=missing_end,
                                use_cache=True
                            )
                            
                            if not ticks_df.empty:
                                print(f"   Загружено {len(ticks_df):,} тиков")
                                candles_from_ticks = loader.create_minute_candles_from_ticks(ticks_df)
                                if not candles_from_ticks.empty:
                                    print(f"   Создано {len(candles_from_ticks)} минутных свечей из тиков (период ДО)")
                                    candles_to_add.append(candles_from_ticks)
                    
                    # Загружаем тики за период ПОСЛЕ доступных данных
                    if actual_end < end_date:
                        missing_start = actual_end + timedelta(minutes=1)
                        missing_end = end_date
                        
                        if missing_start < missing_end:
                            print(f"   Загрузка тиков за период ПОСЛЕ доступных данных: {missing_start.strftime('%Y-%m-%d %H:%M')} - {missing_end.strftime('%Y-%m-%d %H:%M')}...")
                            ticks_df = tick_loader.load_ticks(
                                symbol=symbol,
                                start_time=missing_start,
                                end_time=missing_end,
                                use_cache=True
                            )
                            
                            if not ticks_df.empty:
                                print(f"   Загружено {len(ticks_df):,} тиков")
                                candles_from_ticks = loader.create_minute_candles_from_ticks(ticks_df)
                                if not candles_from_ticks.empty:
                                    print(f"   Создано {len(candles_from_ticks)} минутных свечей из тиков (период ПОСЛЕ)")
                                    candles_to_add.append(candles_from_ticks)
                    
                    # Объединяем все данные
                    if candles_to_add:
                        # Объединяем: свечи из тиков (до), данные из MT5, свечи из тиков (после)
                        all_candles = candles_to_add + [df]
                        df = pd.concat(all_candles).sort_index()
                        
                        # Удаляем дубликаты (приоритет данным из MT5 - keep='last' сохраняет последний, т.е. из MT5)
                        df = df[~df.index.duplicated(keep='last')]
                        
                        print(f"   ✓ Итого минутных свечей: {len(df)}")
                        print(f"   Период: {df.index.min().strftime('%Y-%m-%d %H:%M')} - {df.index.max().strftime('%Y-%m-%d %H:%M')}")
                    else:
                        print(f"   ⚠️  Тики за недостающие периоды не найдены. Используем только доступные данные.")
                
                if not df.empty:
                    print(f"   ✓ Загружено {len(df)} минутных свечей за период {df.index.min().strftime('%Y-%m-%d %H:%M')} - {df.index.max().strftime('%Y-%m-%d %H:%M')}")
                    return df
            
            # Если данных нет вообще, пытаемся создать из тиков
            if df.empty and use_ticks_fallback:
                print(f"   ⚠️  Минутные данные не найдены. Пытаемся создать из тиков...")
                
                from data.tick_data_loader import TickDataLoader
                # Создаем TickDataLoader с использованием существующего подключения MT5
                tick_loader = TickDataLoader(mt5_connection=loader, use_cache=True, offline_mode=self.offline_mode)
                
                print(f"   Загрузка тиков за период {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}...")
                ticks_df = tick_loader.load_ticks(
                    symbol=symbol,
                    start_time=start_date,
                    end_time=end_date,
                    use_cache=True
                )
                
                if not ticks_df.empty:
                    print(f"   Загружено {len(ticks_df):,} тиков")
                    print(f"   Создание минутных свечей из тиков...")
                    
                    df = loader.create_minute_candles_from_ticks(ticks_df)
                    
                    if not df.empty:
                        print(f"   ✓ Создано {len(df)} минутных свечей из тиков")
                        print(f"   Период: {df.index.min().strftime('%Y-%m-%d %H:%M')} - {df.index.max().strftime('%Y-%m-%d %H:%M')}")
                        return df
            
            # Если ничего не получилось
            if df.empty:
                error_msg = f"Не удалось загрузить данные для {symbol}"
                error_msg += f"\n   Период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}"
                error_msg += f"\n   Проверьте:"
                error_msg += f"\n   1. Символ {symbol} доступен в вашем брокере"
                error_msg += f"\n   2. MT5 терминал запущен и подключен к серверу"
                error_msg += f"\n   3. Символ добавлен в Market Watch"
                error_msg += f"\n   4. Есть исторические данные (минутные или тиковые) за указанный период"
                if available_symbols and symbol not in available_symbols:
                    similar = [s for s in available_symbols if 'GOLD' in s.upper() or 'XAU' in s.upper()]
                    if similar:
                        error_msg += f"\n   5. Попробуйте использовать: --symbol {similar[0]}"
                raise ValueError(error_msg)
            
            return df
        
        finally:
            # Отключаемся от MT5 только после завершения всех операций
            # Если tick_loader использовался, он мог создать свое подключение,
            # но мы все равно отключаем основное подключение loader
            loader.disconnect()
    
    def load_higher_timeframes(self,
                              symbol: str = 'XAUUSD',
                              end_date: Optional[datetime] = None,
                              months: Optional[int] = None,
                              days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Загружает данные старших таймфреймов
        
        Args:
            symbol: Символ
            end_date: Конечная дата
            months: Количество месяцев (если None и days не указан, используется self.training_months)
            days: Количество дней (приоритет над months)
        
        Returns:
            Словарь {timeframe: DataFrame}
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Определяем период: приоритет у days, если указан
        if days is not None:
            period_days = days
        elif months is not None:
            period_days = months * 30
        else:
            months = self.training_months
            period_days = months * 30
        
        start_date = end_date - timedelta(days=period_days)
        
        # В offline режиме создаем старшие таймфреймы из минутных данных
        if self.offline_mode:
            print(f"[OFFLINE MODE] Создание старших таймфреймов из минутных данных...")
            
            # Сначала загружаем минутные данные (в offline режиме они уже загружены из кэша)
            minute_df = self.load_gold_data(
                symbol=symbol,
                end_date=end_date,
                months=months,
                days=days,
                use_ticks_fallback=True  # В offline режиме используем тики из кэша
            )
            
            if minute_df.empty:
                raise ValueError("Не удалось загрузить минутные данные для создания старших таймфреймов")
            
            # Создаем старшие таймфреймы через агрегацию
            higher_timeframes = {}
            
            for tf in self.config.higher_timeframes:
                df_tf = MT5DataLoader.aggregate_timeframe_from_minutes(minute_df, tf)
                if not df_tf.empty:
                    # Фильтруем по нужному периоду
                    df_tf = df_tf[(df_tf.index >= start_date) & (df_tf.index <= end_date)]
                    if not df_tf.empty:
                        higher_timeframes[tf] = df_tf
                        print(f"[{_get_timestamp()}] Создано {len(df_tf)} свечей для таймфрейма {tf} из минутных данных")
            
            return higher_timeframes
        
        # Обычный режим с подключением к MT5
        loader = MT5DataLoader()
        if not loader.connect():
            raise ConnectionError("Не удалось подключиться к MT5")
        
        try:
            higher_timeframes = {}
            for tf in self.config.higher_timeframes:
                df_tf = loader.load_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date
                )
                if not df_tf.empty:
                    higher_timeframes[tf] = df_tf
                    print(f"[{_get_timestamp()}] Загружено {len(df_tf)} свечей для таймфрейма {tf}")
            
            return higher_timeframes
        
        finally:
            loader.disconnect()
    
    def load_tick_data(self, 
                      minute_times: pd.DatetimeIndex,
                      symbol: str = 'XAUUSD',
                      lookback_minutes: int = 1) -> Dict[datetime, pd.DataFrame]:
        """
        Загружает тиковые данные для минутных свечей
        
        Args:
            symbol: Символ
            minute_times: DatetimeIndex с временами минутных свечей
            lookback_minutes: Количество минут тиков для загрузки
        
        Returns:
            Словарь {minute_time: ticks_df}
        """
        tick_loader = TickDataLoader(offline_mode=self.offline_mode)
        
        ticks_data = tick_loader.load_ticks_batch(
            symbol=symbol,
            minute_times=minute_times,
            lookback_minutes=lookback_minutes
        )
        
        print(f"[{_get_timestamp()}] Загружено тиковых данных для {len(ticks_data)} минутных свечей")
        return ticks_data
    
    def prepare_full_dataset(self,
                            symbol: str = 'XAUUSD',
                            end_date: Optional[datetime] = None,
                            months: Optional[int] = None,
                            days: Optional[int] = None,
                            load_ticks: bool = True,
                            load_higher_tf: bool = True,
                            use_cache: bool = True,
                            force_regenerate: bool = False,
                            ask_on_existing: bool = True) -> pd.DataFrame:
        """
        Подготавливает полный датасет для обучения
        
        Args:
            symbol: Символ
            end_date: Конечная дата
            months: Количество месяцев (используется если days не указан)
            days: Количество дней (приоритет над months)
            load_ticks: Загружать ли тиковые данные
            load_higher_tf: Загружать ли старшие таймфреймы
            use_cache: Использовать сохраненный кэш, если он есть
            force_regenerate: Принудительно регенерировать данные (игнорировать кэш)
            ask_on_existing: Спрашивать пользователя, если файл существует
        
        Returns:
            DataFrame со всеми фичами и целевыми переменными
        """
        # Определяем период: приоритет у days, если указан
        if days is not None:
            period_days = days
            period_label = f"{days}d"
        elif months is not None:
            period_days = months * 30
            period_label = f"{months}m"
        else:
            months = self.config.training_data_months
            period_days = months * 30
            period_label = f"{months}m"
        
        # Проверяем наличие сохраненного файла
        cache_file = self._get_cache_file_path(symbol, period_label, end_date, load_ticks, load_higher_tf)
        
        if use_cache and not force_regenerate and cache_file.exists():
            if ask_on_existing:
                print(f"\n[{_get_timestamp()}] Найден сохраненный файл: {cache_file}")
                print(f"[{_get_timestamp()}] Размер файла: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
                print(f"[{_get_timestamp()}] Дата создания: {datetime.fromtimestamp(cache_file.stat().st_mtime)}")
                
                response = input(f"[{_get_timestamp()}] Загрузить сохраненные данные? (y/n): ").strip().lower()
                if response in ['y', 'yes', 'да', 'д', '']:
                    print(f"[{_get_timestamp()}] Загрузка сохраненных данных...")
                    try:
                        df = pd.read_pickle(cache_file)
                        print(f"[{_get_timestamp()}] ✓ Загружено {len(df)} образцов из кэша")
                        print(f"[{_get_timestamp()}] Количество фичей: {len([c for c in df.columns if not c.startswith('future_return') and c != 'signal_class' and c != 'signal_class_name' and c != 'max_future_return'])}")
                        return df
                    except Exception as e:
                        print(f"[{_get_timestamp()}] ⚠ Ошибка при загрузке кэша: {e}")
                        print(f"[{_get_timestamp()}] Продолжаем с генерацией...")
                else:
                    print(f"[{_get_timestamp()}] Продолжаем с генерацией...")
            else:
                # Автоматически загружаем, если не спрашиваем
                print(f"[{_get_timestamp()}] Загрузка сохраненных данных из {cache_file}...")
                try:
                    df = pd.read_pickle(cache_file)
                    print(f"[{_get_timestamp()}] ✓ Загружено {len(df)} образцов из кэша")
                    return df
                except Exception as e:
                    print(f"[{_get_timestamp()}] ⚠ Ошибка при загрузке кэша: {e}")
                    print(f"[{_get_timestamp()}] Продолжаем с генерацией...")
        
        timestamp = _get_timestamp()
        print(f"[{timestamp}] " + "=" * 60)
        print(f"[{timestamp}] Подготовка данных для обучения модели")
        print(f"[{timestamp}] " + "=" * 60)
        
        # 1. Загрузка минутных данных
        print(f"\n[{_get_timestamp()}] 1. Загрузка минутных данных...")
        df_minute = self.load_gold_data(symbol, end_date, months, days)
        
        # 2. Загрузка старших таймфреймов
        higher_timeframes = None
        if load_higher_tf:
            print(f"\n[{_get_timestamp()}] 2. Загрузка старших таймфреймов...")
            higher_timeframes = self.load_higher_timeframes(symbol, end_date, months, days)
        
        # 3. Загрузка тиковых данных
        ticks_data = None
        if load_ticks:
            print(f"\n[{_get_timestamp()}] 3. Загрузка тиковых данных...")
            try:
                ticks_data = self.load_tick_data(
                    minute_times=df_minute.index,
                    symbol=symbol,
                    lookback_minutes=self.config.tick_lookback_minutes
                )
                if len(ticks_data) == 0:
                    print(f"[{_get_timestamp()}]   Предупреждение: тиковые данные не загружены. Продолжаем без них.")
            except Exception as e:
                print(f"[{_get_timestamp()}]   Предупреждение: ошибка при загрузке тиков: {e}")
                print(f"[{_get_timestamp()}]   Продолжаем без тиковых данных.")
                ticks_data = None
        
        # 4. Генерация фичей
        print(f"\n[{_get_timestamp()}] 4. Генерация фичей...")
        df_features = self.feature_engineer.create_features(
            df_minute,
            higher_timeframes_data=higher_timeframes,
            ticks_data=ticks_data,
            add_targets=False,  # Целевые переменные добавим отдельно
            symbol=symbol,
            save_intermediate=True,  # Сохранять промежуточные результаты
            resume=True  # Продолжить с сохраненного прогресса
        )
        
        # 5. Генерация целевых переменных
        print(f"\n[{_get_timestamp()}] 5. Генерация целевых переменных...")
        df_with_targets = self.target_generator.generate_targets(
            df_features,
            price_column='close'
        )
        
        # 6. Анализ распределения классов
        timestamp = _get_timestamp()
        print(f"\n[{timestamp}] 6. Анализ распределения классов:")
        class_dist = self.target_generator.get_class_distribution(df_with_targets)
        print(f"[{timestamp}] {class_dist}")
        print(f"[{timestamp}] Процентное распределение:")
        print(f"[{timestamp}] {(class_dist / len(df_with_targets) * 100).round(2)}")
        
        timestamp = _get_timestamp()
        print(f"\n[{timestamp}] " + "=" * 60)
        print(f"[{timestamp}] Итого подготовлено {len(df_with_targets)} образцов")
        
        # Подсчет фичей
        feature_cols = [c for c in df_with_targets.columns 
                       if not c.startswith('future_return') 
                       and c != 'signal_class' 
                       and c != 'signal_class_name'
                       and c != 'max_future_return']
        print(f"[{timestamp}] Количество фичей: {len(feature_cols)}")
        print(f"[{timestamp}] " + "=" * 60)
        
        # Сохраняем результат в кэш
        if use_cache:
            try:
                cache_file = self._get_cache_file_path(symbol, months, end_date, load_ticks, load_higher_tf)
                df_with_targets.to_pickle(cache_file)
                file_size_mb = cache_file.stat().st_size / 1024 / 1024
                print(f"[{_get_timestamp()}] ✓ Данные сохранены в кэш: {cache_file}")
                print(f"[{_get_timestamp()}] Размер файла: {file_size_mb:.1f} MB")
            except Exception as e:
                print(f"[{_get_timestamp()}] ⚠ Ошибка при сохранении в кэш: {e}")
        
        return df_with_targets
    
    def save_prepared_data(self, df: pd.DataFrame, filepath: str):
        """
        Сохраняет подготовленные данные
        
        Args:
            df: DataFrame с данными
            filepath: Путь для сохранения
        """
        # Создаем директорию, если её нет
        filepath_obj = Path(filepath)
        filepath_obj.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=True)
        print(f"[{_get_timestamp()}] Данные сохранены в {filepath}")
    
    def load_prepared_data(self, filepath: str) -> pd.DataFrame:
        """
        Загружает подготовленные данные
        
        Args:
            filepath: Путь к файлу
        
        Returns:
            DataFrame с данными
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"[{_get_timestamp()}] Загружено {len(df)} образцов из {filepath}")
        return df

