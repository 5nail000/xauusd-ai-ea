"""
Модуль для подготовки данных по золоту (XAUUSD)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict
from data.mt5_data_loader import MT5DataLoader
from data.tick_data_loader import TickDataLoader
from features.feature_engineering import FeatureEngineer
from data.target_generator import TargetGenerator
from config.feature_config import FeatureConfig

class GoldDataPreparator:
    """
    Класс для подготовки данных по золоту для обучения модели
    """
    
    def __init__(self, 
                 config: Optional[FeatureConfig] = None,
                 training_months: int = 6):
        """
        Args:
            config: Конфигурация фичей
            training_months: Количество месяцев данных для обучения (по умолчанию 6)
        """
        self.config = config if config is not None else FeatureConfig()
        self.training_months = training_months
        self.feature_engineer = FeatureEngineer(self.config)
        self.target_generator = TargetGenerator(
            breakout_threshold=50.0,
            bounce_threshold=30.0,
            lookahead_periods=60
        )
    
    def load_gold_data(self, 
                      symbol: str = 'XAUUSD',
                      end_date: Optional[datetime] = None,
                      months: Optional[int] = None) -> pd.DataFrame:
        """
        Загружает минутные данные по золоту
        
        Args:
            symbol: Символ (по умолчанию XAUUSD)
            end_date: Конечная дата (если None, используется текущая дата)
            months: Количество месяцев (если None, используется self.training_months)
        
        Returns:
            DataFrame с минутными данными
        """
        if months is None:
            months = self.training_months
        
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=months * 30)
        
        loader = MT5DataLoader()
        if not loader.connect():
            raise ConnectionError("Не удалось подключиться к MT5")
        
        try:
            df = loader.load_data(
                symbol=symbol,
                timeframe='M1',
                start_date=start_date,
                end_date=end_date
            )
            
            if df.empty:
                raise ValueError(f"Не удалось загрузить данные для {symbol}")
            
            print(f"Загружено {len(df)} минутных свечей за период {start_date} - {end_date}")
            return df
        
        finally:
            loader.disconnect()
    
    def load_higher_timeframes(self, 
                              symbol: str = 'XAUUSD',
                              end_date: Optional[datetime] = None,
                              months: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Загружает данные старших таймфреймов
        
        Args:
            symbol: Символ
            end_date: Конечная дата
            months: Количество месяцев
        
        Returns:
            Словарь {timeframe: DataFrame}
        """
        if months is None:
            months = self.training_months
        
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=months * 30)
        
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
                    print(f"Загружено {len(df_tf)} свечей для таймфрейма {tf}")
            
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
        tick_loader = TickDataLoader()
        
        ticks_data = tick_loader.load_ticks_batch(
            symbol=symbol,
            minute_times=minute_times,
            lookback_minutes=lookback_minutes
        )
        
        print(f"Загружено тиковых данных для {len(ticks_data)} минутных свечей")
        return ticks_data
    
    def prepare_full_dataset(self, 
                            symbol: str = 'XAUUSD',
                            end_date: Optional[datetime] = None,
                            months: Optional[int] = None,
                            load_ticks: bool = True,
                            load_higher_tf: bool = True) -> pd.DataFrame:
        """
        Подготавливает полный датасет для обучения
        
        Args:
            symbol: Символ
            end_date: Конечная дата
            months: Количество месяцев
            load_ticks: Загружать ли тиковые данные
            load_higher_tf: Загружать ли старшие таймфреймы
        
        Returns:
            DataFrame со всеми фичами и целевыми переменными
        """
        print("=" * 60)
        print("Подготовка данных для обучения модели")
        print("=" * 60)
        
        # 1. Загрузка минутных данных
        print("\n1. Загрузка минутных данных...")
        df_minute = self.load_gold_data(symbol, end_date, months)
        
        # 2. Загрузка старших таймфреймов
        higher_timeframes = None
        if load_higher_tf:
            print("\n2. Загрузка старших таймфреймов...")
            higher_timeframes = self.load_higher_timeframes(symbol, end_date, months)
        
        # 3. Загрузка тиковых данных
        ticks_data = None
        if load_ticks:
            print("\n3. Загрузка тиковых данных...")
            try:
                ticks_data = self.load_tick_data(
                    minute_times=df_minute.index,
                    symbol=symbol,
                    lookback_minutes=self.config.tick_lookback_minutes
                )
                if len(ticks_data) == 0:
                    print("  Предупреждение: тиковые данные не загружены. Продолжаем без них.")
            except Exception as e:
                print(f"  Предупреждение: ошибка при загрузке тиков: {e}")
                print("  Продолжаем без тиковых данных.")
                ticks_data = None
        
        # 4. Генерация фичей
        print("\n4. Генерация фичей...")
        df_features = self.feature_engineer.create_features(
            df_minute,
            higher_timeframes_data=higher_timeframes,
            ticks_data=ticks_data,
            add_targets=False  # Целевые переменные добавим отдельно
        )
        
        # 5. Генерация целевых переменных
        print("\n5. Генерация целевых переменных...")
        df_with_targets = self.target_generator.generate_targets(
            df_features,
            price_column='close'
        )
        
        # 6. Анализ распределения классов
        print("\n6. Анализ распределения классов:")
        class_dist = self.target_generator.get_class_distribution(df_with_targets)
        print(class_dist)
        print(f"\nПроцентное распределение:")
        print((class_dist / len(df_with_targets) * 100).round(2))
        
        print("\n" + "=" * 60)
        print(f"Итого подготовлено {len(df_with_targets)} образцов")
        
        # Подсчет фичей
        feature_cols = [c for c in df_with_targets.columns 
                       if not c.startswith('future_return') 
                       and c != 'signal_class' 
                       and c != 'signal_class_name'
                       and c != 'max_future_return']
        print(f"Количество фичей: {len(feature_cols)}")
        print("=" * 60)
        
        return df_with_targets
    
    def save_prepared_data(self, df: pd.DataFrame, filepath: str):
        """
        Сохраняет подготовленные данные
        
        Args:
            df: DataFrame с данными
            filepath: Путь для сохранения
        """
        df.to_csv(filepath, index=True)
        print(f"Данные сохранены в {filepath}")
    
    def load_prepared_data(self, filepath: str) -> pd.DataFrame:
        """
        Загружает подготовленные данные
        
        Args:
            filepath: Путь к файлу
        
        Returns:
            DataFrame с данными
        """
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        print(f"Загружено {len(df)} образцов из {filepath}")
        return df

