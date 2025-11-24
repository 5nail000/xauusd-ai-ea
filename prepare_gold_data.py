"""
Скрипт для подготовки данных по золоту для обучения модели
"""
import pandas as pd
from datetime import datetime, timedelta
from data.gold_data_prep import GoldDataPreparator
from data.data_splitter import DataSplitter
from config.feature_config import FeatureConfig

def main():
    """
    Подготовка данных по золоту для обучения Transformer модели
    """
    print("=" * 60)
    print("Подготовка данных по золоту (XAUUSD)")
    print("=" * 60)
    
    # Конфигурация
    config = FeatureConfig()
    
    # Количество месяцев данных (можно изменить)
    training_months = 6  # По умолчанию 6 месяцев, можно указать любое количество
    
    # Создаем подготовщик данных
    preparator = GoldDataPreparator(
        config=config,
        training_months=training_months
    )
    
    # Подготавливаем полный датасет
    df = preparator.prepare_full_dataset(
        symbol='XAUUSD',
        end_date=None,  # До текущей даты
        months=training_months,
        load_ticks=True,
        load_higher_tf=True
    )
    
    # Сохраняем подготовленные данные
    output_file = f'data/gold_data_{training_months}months.csv'
    preparator.save_prepared_data(df, output_file)
    
    # Разделяем на train/validation/test
    print("\n" + "=" * 60)
    print("Разделение данных на train/validation/test")
    print("=" * 60)
    
    splitter = DataSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        temporal_split=True  # Временное разделение (без перемешивания)
    )
    
    train_df, val_df, test_df = splitter.split(df, target_column='signal_class')
    
    # Анализ распределения классов
    print("\nРаспределение классов в train:")
    print(splitter.get_class_distribution(train_df))
    
    print("\nРаспределение классов в validation:")
    print(splitter.get_class_distribution(val_df))
    
    print("\nРаспределение классов в test:")
    print(splitter.get_class_distribution(test_df))
    
    # Сохраняем разделенные данные
    train_df.to_csv('data/gold_train.csv', index=True)
    val_df.to_csv('data/gold_val.csv', index=True)
    test_df.to_csv('data/gold_test.csv', index=True)
    
    print("\n" + "=" * 60)
    print("Данные сохранены:")
    print(f"  - Полный датасет: {output_file}")
    print(f"  - Train: data/gold_train.csv")
    print(f"  - Validation: data/gold_val.csv")
    print(f"  - Test: data/gold_test.csv")
    print("=" * 60)

if __name__ == '__main__':
    main()

