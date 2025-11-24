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
    оставляя более простые и базовые
    
    Args:
        high_corr_pairs: Список высококоррелированных пар
        feature_columns: Все фичи
    
    Returns:
        Множество фичей для удаления
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
        default='data/gold_train.csv',
        help='Путь к файлу с данными (по умолчанию: data/gold_train.csv)'
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
        default='data/gold_train_no_corr.csv',
        help='Путь для сохранения данных без коррелированных фичей (по умолчанию: data/gold_train_no_corr.csv)'
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
        and df[col].dtype in [np.number, 'float64', 'int64']
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
    
    # Выбор фичей для удаления
    if args.remove and len(high_corr_pairs) > 0:
        print("\n5. Выбор фичей для удаления...")
        features_to_remove = select_features_to_remove(high_corr_pairs, feature_columns)
        print(f"   Будет удалено {len(features_to_remove)} фичей:")
        for feat in sorted(features_to_remove):
            print(f"     - {feat}")
        
        # Удаление фичей
        print(f"\n6. Удаление фичей из данных...")
        df_cleaned = df.drop(columns=list(features_to_remove))
        
        # Сохранение
        print(f"\n7. Сохранение очищенных данных в {args.output}...")
        df_cleaned.to_csv(args.output)
        print(f"   ✓ Сохранено {len(df_cleaned)} строк, {len(df_cleaned.columns)} колонок")
        print(f"   Удалено {len(features_to_remove)} фичей")
        print(f"   Осталось {len(df_cleaned.columns) - len([c for c in df_cleaned.columns if any(p in c for p in exclude_patterns)])} фичей")
    
    # Построение графика
    if args.plot:
        print("\n8. Построение графика корреляционной матрицы...")
        plot_path = args.input.replace('.csv', '_correlation_matrix.png')
        plot_correlation_matrix(df, feature_columns[:50], save_path=plot_path)
    
    # Статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА")
    print("=" * 80)
    print(f"Всего фичей: {len(feature_columns)}")
    print(f"Высококоррелированных пар (>{args.threshold}): {len(high_corr_pairs)}")
    if args.remove and len(high_corr_pairs) > 0:
        print(f"Удалено фичей: {len(features_to_remove)}")
        print(f"Осталось фичей: {len(feature_columns) - len(features_to_remove)}")
    print("=" * 80)
    
    # Рекомендации
    if len(high_corr_pairs) > 0:
        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("   1. Рассмотрите возможность удаления высококоррелированных фичей")
        print("   2. Запустите с флагом --remove для автоматического удаления")
        print("   3. Проверьте результаты на тестовой выборке")
        print("   4. Используйте --plot для визуализации корреляций")

if __name__ == '__main__':
    main()

