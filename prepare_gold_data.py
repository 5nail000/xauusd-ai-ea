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
  
  # Для экспериментов: случайное разделение со стратификацией
  python prepare_gold_data.py --no-temporal-split
  
  # Для реальной торговли: временное разделение с балансировкой классов
  python prepare_gold_data.py --balance-classes --balance-method undersample
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
    
    parser.add_argument(
        '--no-temporal-split',
        action='store_true',
        help='Использовать случайное разделение со стратификацией вместо временного (для экспериментов)'
    )
    
    parser.add_argument(
        '--balance-classes',
        action='store_true',
        help='Балансировать классы после разделения (только для temporal_split, уменьшает дисбаланс)'
    )
    
    parser.add_argument(
        '--balance-method',
        type=str,
        default='undersample',
        choices=['undersample', 'oversample'],
        help='Метод балансировки классов: undersample (уменьшить большие классы) или oversample (увеличить малые) (по умолчанию: undersample)'
    )
    
    parser.add_argument(
        '--apply-features-exclusions',
        action='store_true',
        help='Применять список исключений из excluded_features.txt при генерации всех фичей'
    )
    
    parser.add_argument(
        '--use-included-features',
        action='store_true',
        help='Использовать только фичи из белого списка included_features.txt (если файл существует и не пуст)'
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
    print(f"  Разделение: {'Случайное (со стратификацией)' if args.no_temporal_split else 'Временное'}")
    if not args.no_temporal_split:
        print(f"  Балансировка классов: {'Да' if args.balance_classes else 'Нет'}")
        if args.balance_classes:
            print(f"  Метод балансировки: {args.balance_method}")
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
        ask_on_existing=not args.no_ask,
        apply_features_exclusions=args.apply_features_exclusions,
        use_included_features=args.use_included_features
    )
    
    # Сохраняем подготовленные данные
    period_label = f"{args.days}d" if args.days is not None else f"{args.months or 6}m"
    output_file = f'workspace/prepared/features/gold_data_{period_label}.csv'
    preparator.save_prepared_data(df, output_file)
    
    # Разделяем на train/validation/test
    print("\n" + "=" * 60)
    print("Разделение данных на train/validation/test")
    print("=" * 60)
    
    use_temporal_split = not args.no_temporal_split
    
    splitter = DataSplitter(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        temporal_split=use_temporal_split
    )
    
    train_df, val_df, test_df = splitter.split(df, target_column='signal_class')
    
    # Балансировка классов (только для temporal_split)
    if args.balance_classes and use_temporal_split:
        print("\n" + "=" * 60)
        print(f"Балансировка классов (метод: {args.balance_method})")
        print("=" * 60)
        
        from data.target_generator import TargetGenerator
        target_gen = TargetGenerator()
        
        print("\nДо балансировки:")
        print(f"  Train: {len(train_df)} образцов")
        print(f"  Val:   {len(val_df)} образцов")
        print(f"  Test:  {len(test_df)} образцов")
        
        train_df = target_gen.balance_classes(train_df, method=args.balance_method)
        val_df = target_gen.balance_classes(val_df, method=args.balance_method)
        test_df = target_gen.balance_classes(test_df, method=args.balance_method)
        
        print("\nПосле балансировки:")
        print(f"  Train: {len(train_df)} образцов")
        print(f"  Val:   {len(val_df)} образцов")
        print(f"  Test:  {len(test_df)} образцов")
    elif args.balance_classes and not use_temporal_split:
        print("\n⚠️  Предупреждение: балансировка классов не нужна при случайном разделении")
        print("   (стратификация уже обеспечивает одинаковое распределение классов)")
    
    # Анализ распределения классов
    print("\n" + "=" * 60)
    print("Распределение классов")
    print("=" * 60)
    print("\nTrain:")
    print(splitter.get_class_distribution(train_df))
    
    print("\nValidation:")
    print(splitter.get_class_distribution(val_df))
    
    print("\nTest:")
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

