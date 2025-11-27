"""
Скрипт для анализа корреляции фичей и удаления высококоррелированных
"""
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Set

# Защищенные фичи - никогда не удаляются
PROTECTED_FEATURES = ['open', 'high', 'low', 'close']

def find_highly_correlated_pairs(df: pd.DataFrame, 
                                 feature_columns: List[str],
                                 threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Находит пары фичей с высокой корреляцией
    
    Args:
        df: DataFrame с данными
        feature_columns: Список колонок-фичей
        threshold: Порог корреляции (по умолчанию 0.95)
    
    Returns:
        Список кортежей (feature1, feature2, correlation)
    """
    # Вычисляем корреляционную матрицу
    corr_matrix = df[feature_columns].corr()
    
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
    
    # Сортируем по абсолютному значению корреляции
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return high_corr_pairs

def select_features_to_remove(high_corr_pairs: List[Tuple[str, str, float]],
                               feature_columns: List[str]) -> Set[str]:
    """
    Выбирает фичи для удаления из высококоррелированных пар
    
    Стратегия: удаляем более сложные или производные фичи,
    оставляя более простые и базовые. Базовые OHLC цены защищены.
    
    Args:
        high_corr_pairs: Список высококоррелированных пар
        feature_columns: Все фичи
    
    Returns:
        Множество фичей для удаления
    """
    features_to_remove = set()
    
    def is_protected(feature_name: str) -> bool:
        """Проверяет, является ли фича защищенной (базовые OHLC)"""
        return feature_name.lower() in [f.lower() for f in PROTECTED_FEATURES]
    
    def get_priority(feature_name: str) -> int:
        """
        Возвращает приоритет фичи (меньше = выше приоритет)
        Приоритеты:
        0 - защищенные фичи (open, high, low, close)
        1 - простые базовые фичи (sma, ema, rsi, macd, atr, momentum)
        2 - производные фичи (close_rolling_mean, close_momentum, returns_rolling)
        3 - сложные/длинные фичи (price_sma_distance, close_rolling_median, multitimeframe)
        4 - lag фичи (close_lag_1, close_lag_2, etc.)
        """
        feature_lower = feature_name.lower()
        
        # Защищенные фичи - наивысший приоритет
        if is_protected(feature_name):
            return 0
        
        # Простые базовые индикаторы (короткие имена)
        simple_indicators = ['sma', 'ema', 'rsi', 'macd', 'atr', 'momentum', 'std', 'bb_']
        if any(ind in feature_lower for ind in simple_indicators) and len(feature_name) < 15:
            # Проверяем, что это не производная фича
            if 'rolling' not in feature_lower and 'distance' not in feature_lower:
                return 1
        
        # Lag фичи - низкий приоритет (можно удалить часть)
        if 'lag' in feature_lower:
            return 4
        
        # Производные фичи (rolling, distance, etc.)
        if any(x in feature_lower for x in ['rolling', 'distance', 'position', 'zscore', 'percentile']):
            return 3
        
        # Сложные/длинные имена
        if len(feature_name) > 20:
            return 3
        
        # По умолчанию средний приоритет
        return 2
    
    def prefer_simple_name(feat1: str, feat2: str) -> str:
        """
        Выбирает более простое имя из двух
        Предпочтения:
        1. Более короткое имя
        2. Меньше подчеркиваний
        3. Более стандартное название (sma > close_rolling_mean)
        """
        # Если одно имя намного короче
        if len(feat1) < len(feat2) - 3:
            return feat1
        if len(feat2) < len(feat1) - 3:
            return feat2
        
        # Если длины похожи, считаем подчеркивания
        underscores1 = feat1.count('_')
        underscores2 = feat2.count('_')
        if underscores1 < underscores2:
            return feat1
        if underscores2 < underscores1:
            return feat2
        
        # Предпочитаем стандартные названия
        feat1_lower = feat1.lower()
        feat2_lower = feat2.lower()
        
        # sma/ema предпочтительнее close_rolling_mean
        if 'sma_' in feat1_lower or 'ema_' in feat1_lower:
            if 'rolling_mean' in feat2_lower:
                return feat1
        if 'sma_' in feat2_lower or 'ema_' in feat2_lower:
            if 'rolling_mean' in feat1_lower:
                return feat2
        
        # momentum предпочтительнее close_momentum
        if feat1_lower == 'momentum' and 'close_momentum' in feat2_lower:
            return feat1
        if feat2_lower == 'momentum' and 'close_momentum' in feat1_lower:
            return feat2
        
        # price_to предпочтительнее distance_to
        if 'price_to' in feat1_lower and 'distance_to' in feat2_lower:
            return feat1
        if 'price_to' in feat2_lower and 'distance_to' in feat1_lower:
            return feat2
        
        # По умолчанию выбираем более короткое
        return feat1 if len(feat1) <= len(feat2) else feat2
    
    for feat1, feat2, corr in high_corr_pairs:
        # Пропускаем, если одна из фичей уже помечена к удалению
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue
        
        # Защищаем базовые OHLC цены - никогда не удаляем
        if is_protected(feat1):
            features_to_remove.add(feat2)
            continue
        if is_protected(feat2):
            features_to_remove.add(feat1)
            continue
        
        # Для полностью идентичных фичей (corr = 1.0) выбираем более простое имя
        if abs(corr) >= 0.99999:
            preferred = prefer_simple_name(feat1, feat2)
            if preferred == feat1:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
            continue
        
        # Для остальных случаев используем приоритеты
        priority1 = get_priority(feat1)
        priority2 = get_priority(feat2)
        
        if priority1 > priority2:
            features_to_remove.add(feat1)
        elif priority2 > priority1:
            features_to_remove.add(feat2)
        else:
            # Если приоритеты равны, используем предпочтение простых имен
            preferred = prefer_simple_name(feat1, feat2)
            if preferred == feat1:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
    
    return features_to_remove

def plot_correlation_matrix(df: pd.DataFrame, 
                           feature_columns: List[str],
                           save_path: str = None,
                           max_features: int = 50):
    """
    Строит тепловую карту корреляционной матрицы
    
    Args:
        df: DataFrame с данными
        feature_columns: Список фичей для визуализации
        save_path: Путь для сохранения графика
        max_features: Максимальное количество фичей для визуализации
    """
    if len(feature_columns) > max_features:
        print(f"⚠️  Слишком много фичей ({len(feature_columns)}). Визуализируем только первые {max_features}")
        feature_columns = feature_columns[:max_features]
    
    corr_matrix = df[feature_columns].corr()
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, 
                annot=False, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation'})
    plt.title('Correlation Matrix of Features', fontsize=16, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_combined_datasets(train_path: str, val_path: str, test_path: str,
                              threshold: float = 0.95) -> Set[str]:
    """
    Анализирует корреляции на объединенном датасете (train+val+test)
    и возвращает список фичей для удаления
    
    Args:
        train_path: Путь к train CSV
        val_path: Путь к val CSV
        test_path: Путь к test CSV
        threshold: Порог корреляции
    
    Returns:
        Множество фичей для удаления
    """
    print("=" * 80)
    print("АНАЛИЗ КОРРЕЛЯЦИИ НА ОБЪЕДИНЕННОМ ДАТАСЕТЕ")
    print("=" * 80)
    
    # Загружаем все три датасета
    datasets = {}
    for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not Path(path).exists():
            print(f"⚠️  Файл {path} не найден, пропускаем...")
            continue
        print(f"\nЗагрузка {name} данных из {path}...")
        datasets[name] = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"   Загружено {len(datasets[name])} строк, {len(datasets[name].columns)} колонок")
    
    if not datasets:
        print("❌ Ошибка: Не найдено ни одного датасета для анализа")
        return set()
    
    # Объединяем все датасеты
    print("\nОбъединение датасетов для анализа...")
    combined_df = pd.concat(datasets.values(), ignore_index=False)
    print(f"   Объединенный датасет: {len(combined_df)} строк, {len(combined_df.columns)} колонок")
    
    # Выбор фичей (исключаем целевые переменные)
    exclude_patterns = ['future_return', 'signal_class', 'signal_class_name', 'max_future_return']
    feature_columns = [
        col for col in combined_df.columns 
        if not any(pattern in col for pattern in exclude_patterns)
        and pd.api.types.is_numeric_dtype(combined_df[col])
    ]
    
    print(f"   Найдено {len(feature_columns)} фичей для анализа")
    
    # Проверка на NaN
    print("\nПроверка данных...")
    nan_counts = combined_df[feature_columns].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"   ⚠️  Найдено {len(cols_with_nan)} фичей с NaN значениями")
        print(f"   Заполняем NaN медианой...")
        combined_df[feature_columns] = combined_df[feature_columns].fillna(combined_df[feature_columns].median())
    else:
        print("   ✓ NaN значений не найдено")
    
    # Анализ корреляции на объединенном датасете
    print(f"\nАнализ корреляции на объединенном датасете (порог: {threshold})...")
    high_corr_pairs = find_highly_correlated_pairs(combined_df, feature_columns, threshold)
    
    if len(high_corr_pairs) == 0:
        print(f"   ✓ Высококоррелированных пар (>{threshold}) не найдено")
        return set()
    
    print(f"   Найдено {len(high_corr_pairs)} высококоррелированных пар")
    
    # Выбор фичей для удаления
    print("\nВыбор фичей для удаления...")
    print(f"   Защищенные фичи (никогда не удаляются): {', '.join(PROTECTED_FEATURES)}")
    features_to_remove = select_features_to_remove(high_corr_pairs, feature_columns)
    
    print(f"\n✓ Будет удалено {len(features_to_remove)} фичей из всех датасетов:")
    for feat in sorted(features_to_remove):
        print(f"     - {feat}")
    
    return features_to_remove

def main():
    parser = argparse.ArgumentParser(
        description='Анализ корреляции фичей и удаление высококоррелированных',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python analyze_feature_correlation.py                    # Анализ с порогом 0.95
  python analyze_feature_correlation.py --threshold 0.90   # Порог 0.90
  python analyze_feature_correlation.py --remove          # Автоматически удалить
  python analyze_feature_correlation.py --plot            # Построить график
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='workspace/prepared/features/gold_train.csv',
        help='Путь к файлу с данными (по умолчанию: workspace/prepared/features/gold_train.csv)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Порог корреляции для удаления (по умолчанию: 0.95)'
    )
    
    parser.add_argument(
        '--remove',
        action='store_true',
        help='Автоматически удалить высококоррелированные фичи'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Построить график корреляционной матрицы'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='workspace/prepared/features/gold_train_no_corr.csv',
        help='Путь для сохранения данных без коррелированных фичей (по умолчанию: workspace/prepared/features/gold_train_no_corr.csv)'
    )
    
    parser.add_argument(
        '--save-tables',
        action='store_true',
        default=True,
        help='Сохранить результаты анализа в таблицы CSV (по умолчанию: включено)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Директория для сохранения таблиц (по умолчанию: та же, что и входной файл)'
    )
    
    parser.add_argument(
        '--features-to-remove',
        type=str,
        default=None,
        help='Путь к файлу со списком фичей для удаления (CSV с колонкой Feature)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("АНАЛИЗ КОРРЕЛЯЦИИ ФИЧЕЙ")
    print("=" * 80)
    
    # Загрузка данных
    print(f"\n1. Загрузка данных из {args.input}...")
    try:
        df = pd.read_csv(args.input, index_col=0, parse_dates=True)
        print(f"   Загружено {len(df)} строк, {len(df.columns)} колонок")
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {args.input} не найден")
        print("   Сначала запустите: python prepare_gold_data.py")
        return
    
    # Выбор фичей (исключаем целевые переменные)
    print("\n2. Выбор фичей для анализа...")
    exclude_patterns = ['future_return', 'signal_class', 'signal_class_name', 'max_future_return']
    feature_columns = [
        col for col in df.columns 
        if not any(pattern in col for pattern in exclude_patterns)
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    print(f"   Найдено {len(feature_columns)} фичей для анализа")
    
    # Проверка на NaN
    print("\n3. Проверка данных...")
    nan_counts = df[feature_columns].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        print(f"   ⚠️  Найдено {len(cols_with_nan)} фичей с NaN значениями")
        print(f"   Заполняем NaN медианой...")
        df[feature_columns] = df[feature_columns].fillna(df[feature_columns].median())
    else:
        print("   ✓ NaN значений не найдено")
    
    # Определение директории для сохранения результатов
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.input).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Анализ корреляции
    print(f"\n4. Анализ корреляции (порог: {args.threshold})...")
    high_corr_pairs = find_highly_correlated_pairs(df, feature_columns, args.threshold)
    
    if len(high_corr_pairs) == 0:
        print(f"   ✓ Высококоррелированных пар (>{args.threshold}) не найдено")
    else:
        print(f"   Найдено {len(high_corr_pairs)} высококоррелированных пар:")
        print("\n   Топ-20 пар с наивысшей корреляцией:")
        for i, (feat1, feat2, corr) in enumerate(high_corr_pairs[:20], 1):
            print(f"   {i:2d}. {feat1[:40]:40s} <-> {feat2[:40]:40s} : {corr:6.3f}")
        
        if len(high_corr_pairs) > 20:
            print(f"   ... и еще {len(high_corr_pairs) - 20} пар")
        
        # Сохранение высококоррелированных пар в таблицу
        if args.save_tables:
            pairs_df = pd.DataFrame(high_corr_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])
            pairs_df['Abs_Correlation'] = pairs_df['Correlation'].abs()
            pairs_df = pairs_df.sort_values('Abs_Correlation', ascending=False)
            pairs_path = output_dir / f'highly_correlated_pairs_threshold_{args.threshold:.2f}.csv'
            pairs_df.to_csv(pairs_path, index=False)
            print(f"\n   ✓ Таблица высококоррелированных пар сохранена: {pairs_path}")
    
    # Выбор фичей для удаления
    features_to_remove = set()
    
    # Если указан файл со списком фичей для удаления, загружаем его
    if args.features_to_remove and Path(args.features_to_remove).exists():
        print("\n5. Загрузка списка фичей для удаления из файла...")
        remove_list_df = pd.read_csv(args.features_to_remove)
        if 'Feature' in remove_list_df.columns:
            features_to_remove = set(remove_list_df['Feature'].tolist())
            print(f"   Загружено {len(features_to_remove)} фичей для удаления")
        else:
            print("   ⚠️  В файле не найдена колонка 'Feature', используем анализ корреляции")
            args.features_to_remove = None
    
    if args.remove and len(high_corr_pairs) > 0 and not features_to_remove:
        print("\n5. Выбор фичей для удаления...")
        print(f"   Защищенные фичи (никогда не удаляются): {', '.join(PROTECTED_FEATURES)}")
        features_to_remove = select_features_to_remove(high_corr_pairs, feature_columns)
        print(f"   Будет удалено {len(features_to_remove)} фичей:")
        for feat in sorted(features_to_remove):
            print(f"     - {feat}")
        
        # Сохранение списка фичей для удаления в таблицу
        if args.save_tables:
            remove_df = pd.DataFrame({
                'Feature': sorted(features_to_remove),
                'Reason': 'High correlation with other features'
            })
            remove_path = output_dir / f'features_to_remove_threshold_{args.threshold:.2f}.csv'
            remove_df.to_csv(remove_path, index=False)
            print(f"   ✓ Таблица фичей для удаления сохранена: {remove_path}")
        
        # Удаление фичей
        print(f"\n6. Удаление фичей из данных...")
        df_cleaned = df.drop(columns=list(features_to_remove))
        
        # Сохранение
        print(f"\n7. Сохранение очищенных данных в {args.output}...")
        df_cleaned.to_csv(args.output)
        print(f"   ✓ Сохранено {len(df_cleaned)} строк, {len(df_cleaned.columns)} колонок")
        print(f"   Удалено {len(features_to_remove)} фичей")
        print(f"   Осталось {len(df_cleaned.columns) - len([c for c in df_cleaned.columns if any(p in c for p in exclude_patterns)])} фичей")
    elif len(high_corr_pairs) > 0:
        # Если не используется --remove, но есть коррелированные пары, все равно сохраним список потенциальных фичей для удаления
        if args.save_tables:
            print("\n5. Определение потенциальных фичей для удаления...")
            print(f"   Защищенные фичи (никогда не удаляются): {', '.join(PROTECTED_FEATURES)}")
            potential_remove = select_features_to_remove(high_corr_pairs, feature_columns)
            if len(potential_remove) > 0:
                remove_df = pd.DataFrame({
                    'Feature': sorted(potential_remove),
                    'Reason': 'High correlation with other features',
                    'Note': 'Use --remove to actually remove these features'
                })
                remove_path = output_dir / f'potential_features_to_remove_threshold_{args.threshold:.2f}.csv'
                remove_df.to_csv(remove_path, index=False)
                print(f"   ✓ Таблица потенциальных фичей для удаления сохранена: {remove_path}")
    
    # Построение графика
    if args.plot:
        step_num = "8" if args.remove and len(high_corr_pairs) > 0 else "6"
        print(f"\n{step_num}. Построение графика корреляционной матрицы...")
        plot_path = output_dir / f'{Path(args.input).stem}_correlation_matrix.png'
        plot_correlation_matrix(df, feature_columns[:50], save_path=str(plot_path))
    
    # Сохранение статистики в таблицу
    if args.save_tables:
        stats_data = {
            'Metric': [
                'Total Features',
                f'Highly Correlated Pairs (>{args.threshold})',
                'Features to Remove',
                'Features Remaining'
            ],
            'Value': [
                len(feature_columns),
                len(high_corr_pairs),
                len(features_to_remove) if features_to_remove else 0,
                len(feature_columns) - len(features_to_remove) if features_to_remove else len(feature_columns)
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        stats_path = output_dir / f'correlation_analysis_stats_threshold_{args.threshold:.2f}.csv'
        stats_df.to_csv(stats_path, index=False)
    
    # Статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА")
    print("=" * 80)
    print(f"Всего фичей: {len(feature_columns)}")
    print(f"Высококоррелированных пар (>{args.threshold}): {len(high_corr_pairs)}")
    if features_to_remove:
        print(f"Удалено фичей: {len(features_to_remove)}")
        print(f"Осталось фичей: {len(feature_columns) - len(features_to_remove)}")
    print("=" * 80)
    
    if args.save_tables:
        print(f"\n📊 Все таблицы сохранены в: {output_dir}")
    
    # Рекомендации
    if len(high_corr_pairs) > 0:
        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("   1. Рассмотрите возможность удаления высококоррелированных фичей")
        print("   2. Запустите с флагом --remove для автоматического удаления")
        print("   3. Базовые OHLC цены (open, high, low, close) защищены и не будут удалены")
        print("   4. Приоритет отдается простым/коротким именам (sma_5 > close_rolling_mean_5)")
        print("   5. Проверьте результаты на тестовой выборке")
        print("   6. Используйте --plot для визуализации корреляций")

if __name__ == '__main__':
    main()

