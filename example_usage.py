"""
Пример использования модулей для генерации фичей
"""
import pandas as pd
from datetime import datetime, timedelta
from data.mt5_data_loader import MT5DataLoader
from data.tick_data_loader import TickDataLoader
from features.feature_engineering import FeatureEngineer
from utils.normalization import FeatureScaler

def main():
    """
    Пример загрузки данных и генерации фичей
    """
    # 1. Загрузка данных из MT5
    print("=== Загрузка данных из MetaTrader 5 ===")
    loader = MT5DataLoader()
    
    if not loader.connect():
        print("Не удалось подключиться к MT5. Используйте демо-данные.")
        # Создаем демо-данные для примера
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')
        df = pd.DataFrame({
            'open': 1.1000 + pd.Series(range(1000)) * 0.0001,
            'high': 1.1005 + pd.Series(range(1000)) * 0.0001,
            'low': 1.0995 + pd.Series(range(1000)) * 0.0001,
            'close': 1.1002 + pd.Series(range(1000)) * 0.0001,
            'volume': 1000 + pd.Series(range(1000)) % 500
        }, index=dates)
    else:
        # Загрузка реальных данных
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 30 дней данных
        
        df = loader.load_data(
            symbol='EURUSD',
            timeframe='M1',
            start_date=start_date,
            end_date=end_date
        )
        
        # Загрузка данных для старших таймфреймов
        higher_timeframes = {}
        for tf in ['H1', 'H4', 'D1']:
            df_tf = loader.load_data(
                symbol='EURUSD',
                timeframe=tf,
                start_date=start_date,
                end_date=end_date
            )
            if not df_tf.empty:
                higher_timeframes[tf] = df_tf
        
        # Загрузка тиковых данных
        print("\n=== Загрузка тиковых данных ===")
        tick_loader = TickDataLoader()
        ticks_data = tick_loader.load_ticks_batch(
            symbol='EURUSD',
            minute_times=df.index,
            lookback_minutes=2  # Последние 2 минуты тиков
        )
        print(f"Загружено тиковых данных для {len(ticks_data)} минутных свечей")
        
        loader.disconnect()
    
    if df.empty:
        print("Ошибка: не удалось загрузить данные")
        return
    
    print(f"Загружено {len(df)} строк данных")
    print(f"Колонки: {df.columns.tolist()}")
    
    # 2. Генерация фичей
    print("\n=== Генерация фичей ===")
    feature_engineer = FeatureEngineer()
    
    # Если есть данные старших таймфреймов и тиковые данные, передаем их
    if 'higher_timeframes' in locals() and higher_timeframes and 'ticks_data' in locals():
        df_features = feature_engineer.create_features(
            df, 
            higher_timeframes_data=higher_timeframes,
            ticks_data=ticks_data,
            add_targets=True
        )
    elif 'higher_timeframes' in locals() and higher_timeframes:
        df_features = feature_engineer.create_features(
            df, 
            higher_timeframes_data=higher_timeframes,
            add_targets=True
        )
    elif 'ticks_data' in locals() and ticks_data:
        df_features = feature_engineer.create_features(
            df,
            ticks_data=ticks_data,
            add_targets=True
        )
    else:
        df_features = feature_engineer.create_features(df, add_targets=True)
    
    print(f"\nСоздано фичей: {len(df_features.columns)}")
    print(f"Размер данных: {len(df_features)} строк")
    
    # 3. Нормализация данных
    print("\n=== Нормализация данных ===")
    scaler = FeatureScaler(method='standard')
    
    # Получаем список фичей (исключая целевые переменные)
    feature_columns = feature_engineer.get_feature_list(df_features)
    
    # Масштабируем только фичи
    df_scaled = scaler.fit_transform(df_features[feature_columns])
    df_features[feature_columns] = df_scaled
    
    print("Нормализация завершена")
    
    # 4. Создание последовательностей для Transformer
    print("\n=== Создание последовательностей ===")
    X, y = feature_engineer.create_sequences(
        df_features,
        sequence_length=60,
        target_column='direction_1'
    )
    
    print(f"Размер X: {X.shape}")
    print(f"Размер y: {y.shape}")
    print(f"Уникальные значения в y: {pd.Series(y).value_counts().to_dict()}")
    
    # 5. Сохранение обработанных данных
    print("\n=== Сохранение данных ===")
    df_features.to_csv('data/processed_features.csv')
    print("Данные сохранены в data/processed_features.csv")
    
    # Сохранение scaler
    scaler.save('models/feature_scaler.pkl')
    print("Scaler сохранен в models/feature_scaler.pkl")
    
    print("\n=== Готово! ===")

if __name__ == '__main__':
    main()

