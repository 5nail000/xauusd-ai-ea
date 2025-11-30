"""
Временный скрипт для удаления колонок support/resistance уровней из подготовленных CSV файлов
Позволяет пересчитать эти фичи с улучшенным алгоритмом без полной переподготовки данных

Использование:
    python utils/temp_remove_level_features.py
    python utils/temp_remove_level_features.py --include-fibonacci  # Также удалить Fibonacci фичи
"""
import pandas as pd
import argparse
from pathlib import Path
from typing import List


# Список всех колонок, связанных с support/resistance уровнями
SUPPORT_RESISTANCE_COLUMNS = [
    'support_level',
    'resistance_level',
    'support_width',
    'resistance_width',
    'support_strength',
    'resistance_strength',
    'distance_to_support_sigma',
    'distance_to_resistance_sigma',
    'distance_to_support_atr',
    'distance_to_resistance_atr',
    'distance_to_support_pct',
    'distance_to_resistance_pct',
    'in_support_zone',
    'in_resistance_zone',
    'proximity_to_support',
    'proximity_to_resistance',
    'proximity_to_support_atr',
    'proximity_to_resistance_atr',
    'price_to_support_ratio',
    'price_to_resistance_ratio',
]

# Список колонок, связанных с Fibonacci уровнями
FIBONACCI_COLUMNS = [
    'swing_high',
    'swing_low',
    'swing_range',
    'fib_level_0',
    'fib_level_236',
    'fib_level_382',
    'fib_level_500',
    'fib_level_618',
    'fib_level_786',
    'fib_level_100',
    'distance_to_fib_0',
    'distance_to_fib_236',
    'distance_to_fib_382',
    'distance_to_fib_500',
    'distance_to_fib_618',
    'distance_to_fib_786',
    'distance_to_fib_100',
    'distance_to_fib_0_atr',
    'distance_to_fib_236_atr',
    'distance_to_fib_382_atr',
    'distance_to_fib_500_atr',
    'distance_to_fib_618_atr',
    'distance_to_fib_786_atr',
    'distance_to_fib_100_atr',
    'near_fib_0',
    'near_fib_236',
    'near_fib_382',
    'near_fib_500',
    'near_fib_618',
    'near_fib_786',
    'near_fib_100',
    'current_fib_level',
]


def find_level_columns(df: pd.DataFrame, include_fibonacci: bool = False) -> List[str]:
    """
    Находит все колонки, связанные с уровнями, которые присутствуют в DataFrame
    
    Args:
        df: DataFrame для проверки
        include_fibonacci: Включать ли Fibonacci колонки
    
    Returns:
        Список колонок для удаления
    """
    columns_to_remove = []
    
    # Проверяем support/resistance колонки
    for col in SUPPORT_RESISTANCE_COLUMNS:
        if col in df.columns:
            columns_to_remove.append(col)
    
    # Проверяем Fibonacci колонки (если нужно)
    if include_fibonacci:
        for col in FIBONACCI_COLUMNS:
            if col in df.columns:
                columns_to_remove.append(col)
    
    # Также ищем колонки, которые начинаются с этих префиксов (на случай других вариантов)
    for col in df.columns:
        if col.startswith('support_') or col.startswith('resistance_'):
            if col not in columns_to_remove:
                columns_to_remove.append(col)
        if include_fibonacci and (col.startswith('fib_') or col.startswith('swing_')):
            if col not in columns_to_remove:
                columns_to_remove.append(col)
    
    return columns_to_remove


def remove_level_features_from_file(file_path: Path, include_fibonacci: bool = False, backup: bool = True) -> bool:
    """
    Удаляет колонки уровней из CSV файла
    
    Args:
        file_path: Путь к CSV файлу
        include_fibonacci: Включать ли Fibonacci колонки
        backup: Создавать ли резервную копию
    
    Returns:
        True если успешно, False в противном случае
    """
    if not file_path.exists():
        print(f"⚠ Файл не найден: {file_path}")
        return False
    
    try:
        # Загружаем CSV
        print(f"\n📂 Обработка: {file_path.name}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        original_columns = len(df.columns)
        print(f"  Исходное количество колонок: {original_columns}")
        
        # Находим колонки для удаления
        columns_to_remove = find_level_columns(df, include_fibonacci)
        
        if not columns_to_remove:
            print(f"  ✓ Колонки уровней не найдены, файл не изменен")
            return True
        
        print(f"  Найдено колонок для удаления: {len(columns_to_remove)}")
        print(f"  Колонки: {', '.join(columns_to_remove[:5])}{'...' if len(columns_to_remove) > 5 else ''}")
        
        # Создаем резервную копию
        if backup:
            backup_path = file_path.with_suffix('.csv.backup')
            df.to_csv(backup_path)
            print(f"  ✓ Резервная копия создана: {backup_path.name}")
        
        # Удаляем колонки
        df = df.drop(columns=columns_to_remove)
        
        new_columns = len(df.columns)
        print(f"  Новое количество колонок: {new_columns}")
        print(f"  Удалено колонок: {original_columns - new_columns}")
        
        # Сохраняем обратно
        df.to_csv(file_path)
        print(f"  ✓ Файл обновлен: {file_path.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Ошибка при обработке {file_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Удаление колонок support/resistance уровней из подготовленных CSV файлов',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python utils/temp_remove_level_features.py
  python utils/temp_remove_level_features.py --include-fibonacci
  python utils/temp_remove_level_features.py --no-backup
        """
    )
    
    parser.add_argument(
        '--include-fibonacci',
        action='store_true',
        help='Также удалить Fibonacci колонки'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Не создавать резервные копии файлов'
    )
    
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=None,
        help='Конкретные файлы для обработки (по умолчанию: все gold_*.csv в workspace/prepared/features/)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("УДАЛЕНИЕ КОЛОНОК УРОВНЕЙ ИЗ ПОДГОТОВЛЕННЫХ CSV ФАЙЛОВ")
    print("=" * 80)
    print(f"Включать Fibonacci: {'Да' if args.include_fibonacci else 'Нет'}")
    print(f"Создавать резервные копии: {'Нет' if args.no_backup else 'Да'}")
    print("=" * 80)
    
    # Определяем файлы для обработки
    features_dir = Path('workspace/prepared/features')
    
    if args.files:
        # Обрабатываем указанные файлы
        files_to_process = []
        for file_str in args.files:
            file_path = Path(file_str)
            if not file_path.is_absolute():
                file_path = features_dir / file_path
            if file_path.exists():
                files_to_process.append(file_path)
            else:
                print(f"⚠ Файл не найден: {file_path}")
    else:
        # Ищем все gold_*.csv файлы
        files_to_process = list(features_dir.glob('gold_*.csv'))
    
    if not files_to_process:
        print(f"\n❌ Не найдено файлов для обработки в {features_dir}")
        print("   Убедитесь, что файлы gold_train.csv, gold_val.csv, gold_test.csv существуют")
        return
    
    print(f"\nНайдено файлов для обработки: {len(files_to_process)}")
    
    # Обрабатываем каждый файл
    success_count = 0
    for file_path in files_to_process:
        if remove_level_features_from_file(file_path, args.include_fibonacci, not args.no_backup):
            success_count += 1
    
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"Успешно обработано: {success_count}/{len(files_to_process)}")
    
    if success_count == len(files_to_process):
        print("\n✓ Все файлы успешно обновлены!")
        print("\nСледующие шаги:")
        print("1. Запустите: python prepare_gold_data.py --months 6")
        print("2. Система автоматически пересчитает только фичи уровней")
        print("3. Остальные фичи останутся без изменений")
    else:
        print("\n⚠ Некоторые файлы не были обработаны. Проверьте ошибки выше.")


if __name__ == '__main__':
    main()

