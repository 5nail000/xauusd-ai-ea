"""
Скрипт для подготовки данных по золоту для обучения модели
"""
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from data.gold_data_prep import GoldDataPreparator
from data.data_splitter import DataSplitter
from config.feature_config import FeatureConfig

def main():
    """
    Подготовка данных по золоту для обучения Transformer модели
    """
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Подготовка данных по золоту для обучения Transformer модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python prepare_gold_data.py                    # 6 месяцев (по умолчанию)
  python prepare_gold_data.py --months 12         # 12 месяцев
  python prepare_gold_data.py --days 30            # 30 дней
  python prepare_gold_data.py -d 7                # 7 дней
  python prepare_gold_data.py -m 12 --no-ticks    # 12 месяцев без тиков
  python prepare_gold_data.py -m 6 --force       # Принудительная регенерация
        """
    )
    
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=None,
        help='Количество месяцев данных для обучения (по умолчанию: 6, если --days не указан)'
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=None,
        help='Количество дней данных для обучения (приоритет над --months)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='XAUUSD',
        help='Торговый символ (по умолчанию: XAUUSD)'
    )
    
    parser.add_argument(
        '--no-ticks',
        action='store_true',
        help='Не загружать тиковые данные'
    )
    
    parser.add_argument(
        '--no-higher-tf',
        action='store_true',
        help='Не загружать старшие таймфреймы'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Принудительно регенерировать данные (игнорировать кэш)'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Не использовать кэш (не сохранять и не загружать)'
    )
    
    parser.add_argument(
        '--no-ask',
        action='store_true',
        help='Не спрашивать при наличии сохраненных данных (автоматически загружать)'
    )
    
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Режим offline - работа только с кэшированными данными без подключения к MT5'
    )
    
    args = parser.parse_args()
    
    # Определяем период для отображения
    if args.days is not None:
        period_str = f"{args.days} дней"
        months_for_display = None
    elif args.months is not None:
        period_str = f"{args.months} месяцев"
        months_for_display = args.months
    else:
        period_str = "6 месяцев (по умолчанию)"
        months_for_display = 6
    
    print("=" * 60)
    print("Подготовка данных по золоту (XAUUSD)")
    print("=" * 60)
    print(f"Параметры:")
    print(f"  Период данных: {period_str}")
    print(f"  Символ: {args.symbol}")
    print(f"  Тики: {'Нет' if args.no_ticks else 'Да'}")
    print(f"  Старшие таймфреймы: {'Нет' if args.no_higher_tf else 'Да'}")
    print(f"  Принудительная регенерация: {'Да' if args.force else 'Нет'}")
    print(f"  Использование кэша: {'Нет' if args.no_cache else 'Да'}")
    print(f"  Режим offline: {'Да' if args.offline else 'Нет'}")
    print("=" * 60)
    
    # Конфигурация
    config = FeatureConfig()
    
    # Создаем подготовщик данных
    preparator = GoldDataPreparator(
        config=config,
        training_months=args.months,
        offline_mode=args.offline
    )
    
    # Подготавливаем полный датасет
    df = preparator.prepare_full_dataset(
        symbol=args.symbol,
        end_date=None,  # До текущей даты
        months=args.months if args.months is not None else (6 if args.days is None else None),
        days=args.days,
        load_ticks=not args.no_ticks,
        load_higher_tf=not args.no_higher_tf,
        use_cache=not args.no_cache,
        force_regenerate=args.force,
        ask_on_existing=not args.no_ask
    )
    
    # Сохраняем подготовленные данные
    period_label = f"{args.days}d" if args.days is not None else f"{args.months or 6}m"
    output_file = f'workspace/prepared/features/gold_data_{period_label}.csv'
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
    # Создаем директорию, если её нет
    features_dir = Path('workspace/prepared/features')
    features_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv('workspace/prepared/features/gold_train.csv', index=True)
    val_df.to_csv('workspace/prepared/features/gold_val.csv', index=True)
    test_df.to_csv('workspace/prepared/features/gold_test.csv', index=True)
    
    print("\n" + "=" * 60)
    print("Данные сохранены:")
    print(f"  - Полный датасет: {output_file}")
    print(f"  - Train: workspace/prepared/features/gold_train.csv")
    print(f"  - Validation: workspace/prepared/features/gold_val.csv")
    print(f"  - Test: workspace/prepared/features/gold_test.csv")
    print("=" * 60)

if __name__ == '__main__':
    main()

