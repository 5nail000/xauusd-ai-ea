"""
Объединенный скрипт для анализа фичей и формирования списка исключений
Объединяет анализ корреляции и комплексный анализ фичей
Группирует фичи по причинам исключения с комментариями
"""
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Set, Dict
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования (должно быть до импортов, которые могут использовать logger)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Определяем PROTECTED_FEATURES на случай, если импорт не удался
PROTECTED_FEATURES = ['open', 'high', 'low', 'close']

# Импорты из существующих скриптов (опциональные)
try:
    from analyze_feature_correlation import (
        find_highly_correlated_pairs,
        select_features_to_remove,
        PROTECTED_FEATURES as CORR_PROTECTED,
        analyze_combined_datasets
    )
    PROTECTED_FEATURES = CORR_PROTECTED
except ImportError as e:
    logger.warning(f"Не удалось импортировать analyze_feature_correlation: {e}")
    logger.warning("  Функции анализа корреляции будут недоступны")
    
    def find_highly_correlated_pairs(*args, **kwargs):
        raise NotImplementedError("analyze_feature_correlation не доступен")
    
    def select_features_to_remove(*args, **kwargs):
        raise NotImplementedError("analyze_feature_correlation не доступен")
    
    def analyze_combined_datasets(*args, **kwargs):
        raise NotImplementedError("analyze_feature_correlation не доступен")

try:
    from analyze_features_comprehensive import (
        compute_basic_statistics,
        analyze_feature_importance,
        analyze_outliers
    )
except ImportError as e:
    logger.warning(f"Не удалось импортировать analyze_features_comprehensive: {e}")
    logger.warning("  Функции комплексного анализа будут недоступны")
    
    def compute_basic_statistics(*args, **kwargs):
        raise NotImplementedError("analyze_features_comprehensive не доступен")
    
    def analyze_feature_importance(*args, **kwargs):
        raise NotImplementedError("analyze_features_comprehensive не доступен")
    
    def analyze_outliers(*args, **kwargs):
        raise NotImplementedError("analyze_features_comprehensive не доступен")


def find_data_leakage_features(all_features: List[str]) -> List[str]:
    """
    Находит фичи с data leakage (содержат информацию о будущем)
    
    Args:
        all_features: Список всех фичей
    
    Returns:
        Список фичей с data leakage
    """
    data_leakage_patterns = [
        'future_return',
        'max_future_return',
        'direction_',
        'future_volatility_'
    ]
    
    excluded = []
    for feature in all_features:
        for pattern in data_leakage_patterns:
            if pattern in feature.lower():
                excluded.append(feature)
                break
    
    return sorted(list(set(excluded)))


def find_zero_features(stats_df: pd.DataFrame) -> List[str]:
    """
    Находит фичи с 100% нулей
    
    Args:
        stats_df: DataFrame со статистикой фичей
    
    Returns:
        Список фичей с 100% нулей
    """
    excluded = []
    
    if 'zeros' in stats_df.columns and 'count' in stats_df.columns:
        for _, row in stats_df.iterrows():
            feature = row['feature']
            count = row.get('count', 0)
            zeros = row.get('zeros', 0)
            zeros_pct = row.get('zeros_pct', 0)
            
            # Проверяем 100% нулей
            if count > 0 and (zeros == count or zeros_pct >= 99.99):
                excluded.append(feature)
    
    return sorted(excluded)


def find_high_missing_features(stats_df: pd.DataFrame, threshold: float = 90.0) -> List[str]:
    """
    Находит фичи с большим процентом пропусков
    
    Args:
        stats_df: DataFrame со статистикой фичей
        threshold: Порог процента пропусков (по умолчанию 90%)
    
    Returns:
        Список фичей с большим процентом пропусков
    """
    excluded = []
    
    if 'missing_pct' in stats_df.columns:
        for _, row in stats_df.iterrows():
            feature = row['feature']
            missing_pct = row.get('missing_pct', 0)
            
            if pd.notna(missing_pct) and missing_pct > threshold:
                excluded.append(feature)
    
    return sorted(excluded)


def find_low_importance_features(importance_df: pd.DataFrame, 
                                 threshold_percentile: float = 5.0) -> List[str]:
    """
    Находит фичи с низкой важностью (опционально)
    
    Args:
        importance_df: DataFrame с важностью фичей
        threshold_percentile: Процентиль для отсечения (по умолчанию 5%)
    
    Returns:
        Список фичей с низкой важностью
    """
    excluded = []
    
    if 'combined_score' in importance_df.columns and len(importance_df) > 0:
        threshold = importance_df['combined_score'].quantile(threshold_percentile / 100.0)
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            score = row.get('combined_score', 0)
            
            if pd.notna(score) and score <= threshold:
                excluded.append(feature)
    
    return sorted(excluded)


def group_features_by_reason(features_by_reason: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Группирует фичи по причинам исключения, убирая дубликаты
    
    Args:
        features_by_reason: Словарь {причина: [список фичей]}
    
    Returns:
        Словарь с группированными фичами (без дубликатов между группами)
    """
    # Собираем все фичи
    all_excluded = set()
    for features in features_by_reason.values():
        all_excluded.update(features)
    
    # Группируем по приоритету причин (data leakage - самый важный)
    priority_order = [
        'data_leakage',
        'high_correlation',
        'all_zeros',
        'high_missing',
        'low_importance'
    ]
    
    grouped = {}
    used_features = set()
    
    for reason in priority_order:
        if reason not in features_by_reason:
            continue
        
        # Берем только те фичи, которые еще не использованы
        features = [f for f in features_by_reason[reason] if f not in used_features]
        if features:
            grouped[reason] = sorted(features)
            used_features.update(features)
    
    return grouped


def save_excluded_features_grouped(features_by_reason: Dict[str, List[str]],
                                   exclusions_file: Path = None,
                                   correlation_threshold: float = 0.95,
                                   missing_threshold: float = 90.0,
                                   low_importance_percentile: float = 5.0):
    """
    Сохраняет список фичей для исключения с группировкой по причинам
    
    Args:
        features_by_reason: Словарь {причина: [список фичей]}
        exclusions_file: Путь к файлу для сохранения
        correlation_threshold: Порог корреляции (для комментария)
        missing_threshold: Порог пропусков (для комментария)
        low_importance_percentile: Процентиль низкой важности (для комментария)
    """
    if exclusions_file is None:
        exclusions_file = Path('workspace/excluded_features.txt')
    
    # Создаем директорию, если её нет
    exclusions_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Группируем фичи (убираем дубликаты)
    grouped = group_features_by_reason(features_by_reason)
    
    # Названия причин для комментариев
    reason_names = {
        'data_leakage': 'Data Leakage (фичи содержат информацию о будущем)',
        'high_correlation': f'Высокая корреляция (>{correlation_threshold}) с другими фичами',
        'all_zeros': '100% нулевых значений',
        'high_missing': f'Большой процент пропусков (>{missing_threshold}%)',
        'low_importance': f'Низкая важность (нижние {low_importance_percentile}% по combined_score)'
    }
    
    # Сохраняем файл
    try:
        with open(exclusions_file, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("# Список фичей для исключения из обучения\n")
            f.write("# Сгенерировано автоматически на основе анализа корреляции и комплексного анализа\n")
            f.write("# Одна фича на строку\n")
            f.write("# Строки, начинающиеся с #, игнорируются\n")
            f.write("# Пустые строки игнорируются\n")
            f.write("# Фичи сгруппированы по причинам исключения\n\n")
            
            # Группируем и сохраняем
            total_count = 0
            for reason, features in grouped.items():
                if not features:
                    continue
                
                reason_name = reason_names.get(reason, reason)
                f.write(f"# ============================================================\n")
                f.write(f"# {reason_name}\n")
                f.write(f"# Количество: {len(features)}\n")
                f.write(f"# ============================================================\n\n")
                
                for feature in sorted(features):
                    f.write(f"{feature}\n")
                
                f.write("\n")
                total_count += len(features)
            
            # Итоговая статистика
            f.write(f"# ============================================================\n")
            f.write(f"# Всего фичей для исключения: {total_count}\n")
            f.write(f"# ============================================================\n")
        
        logger.info(f"✓ Сохранено {total_count} фичей для исключения в {exclusions_file}")
        logger.info(f"  Групп: {len([g for g in grouped.values() if g])}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка при сохранении файла исключений {exclusions_file}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Объединенный анализ фичей и формирование списка исключений',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовый анализ на объединенном датасете
  python analyze_and_exclude_features.py

  # С настройкой порогов
  python analyze_and_exclude_features.py --correlation-threshold 0.90 --missing-threshold 85

  # Без исключения по низкой важности
  python analyze_and_exclude_features.py --no-low-importance

  # Только анализ корреляции
  python analyze_and_exclude_features.py --only-correlation
        """
    )
    
    parser.add_argument(
        '--train',
        type=str,
        default='workspace/prepared/features/gold_train.csv',
        help='Путь к train CSV (по умолчанию: workspace/prepared/features/gold_train.csv)'
    )
    
    parser.add_argument(
        '--val',
        type=str,
        default='workspace/prepared/features/gold_val.csv',
        help='Путь к val CSV (по умолчанию: workspace/prepared/features/gold_val.csv)'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        default='workspace/prepared/features/gold_test.csv',
        help='Путь к test CSV (по умолчанию: workspace/prepared/features/gold_test.csv)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='signal_class',
        help='Название целевой переменной (по умолчанию: signal_class)'
    )
    
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.95,
        help='Порог корреляции для удаления (по умолчанию: 0.95)'
    )
    
    parser.add_argument(
        '--missing-threshold',
        type=float,
        default=90.0,
        help='Порог процента пропусков для исключения (по умолчанию: 90.0)'
    )
    
    parser.add_argument(
        '--low-importance-percentile',
        type=float,
        default=5.0,
        help='Процентиль для исключения фичей с низкой важностью (по умолчанию: 5.0, 0 = отключить)'
    )
    
    parser.add_argument(
        '--no-low-importance',
        action='store_true',
        help='Не исключать фичи по низкой важности'
    )
    
    parser.add_argument(
        '--only-correlation',
        action='store_true',
        help='Выполнить только анализ корреляции (без комплексного анализа)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='workspace/excluded_features.txt',
        help='Путь для сохранения списка исключений (по умолчанию: workspace/excluded_features.txt)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("ОБЪЕДИНЕННЫЙ АНАЛИЗ ФИЧЕЙ И ФОРМИРОВАНИЕ СПИСКА ИСКЛЮЧЕНИЙ")
    logger.info("=" * 80)
    
    # Проверка файлов
    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    
    if not train_path.exists():
        logger.error(f"❌ Файл не найден: {train_path}")
        logger.error("   Сначала запустите: python prepare_gold_data.py")
        return 1
    
    # Загружаем данные
    logger.info("\n1. Загрузка данных...")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    logger.info(f"   Train: {len(train_df)} образцов, {len(train_df.columns)} колонок")
    
    val_df = None
    if val_path.exists():
        val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
        logger.info(f"   Val: {len(val_df)} образцов, {len(val_df.columns)} колонок")
    
    test_df = None
    if test_path.exists():
        test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
        logger.info(f"   Test: {len(test_df)} образцов, {len(test_df.columns)} колонок")
    
    # Объединяем для анализа
    combined_df = train_df.copy()
    if val_df is not None:
        combined_df = pd.concat([combined_df, val_df])
    if test_df is not None:
        combined_df = pd.concat([combined_df, test_df])
    
    logger.info(f"   Объединенный датасет: {len(combined_df)} образцов")
    
    # Определяем фичи
    exclude_patterns = ['future_return', 'signal_class', 'signal_class_name', 'max_future_return']
    all_features = [
        col for col in combined_df.columns 
        if not any(pattern in col for pattern in exclude_patterns)
        and pd.api.types.is_numeric_dtype(combined_df[col])
    ]
    
    logger.info(f"   Всего фичей для анализа: {len(all_features)}")
    
    # Словарь для группировки фичей по причинам
    features_by_reason = defaultdict(list)
    
    # 1. Data Leakage фичи
    logger.info("\n2. Поиск фичей с data leakage...")
    data_leakage = find_data_leakage_features(all_features)
    features_by_reason['data_leakage'] = data_leakage
    logger.info(f"   ✓ Найдено {len(data_leakage)} фичей с data leakage")
    if data_leakage:
        logger.info(f"   Примеры: {', '.join(data_leakage[:5])}")
    
    # 2. Анализ корреляции
    logger.info(f"\n3. Анализ корреляции (порог: {args.correlation_threshold})...")
    try:
        if val_path.exists() and test_path.exists():
            # Анализ на объединенном датасете
            correlated_features = analyze_combined_datasets(
                str(train_path),
                str(val_path),
                str(test_path),
                threshold=args.correlation_threshold
            )
        else:
            # Анализ только на train
            logger.info("   ⚠️  Val/Test не найдены, анализируем только train")
            feature_columns = [
                col for col in train_df.columns 
                if not any(pattern in col for pattern in exclude_patterns)
                and pd.api.types.is_numeric_dtype(train_df[col])
            ]
            
            # Заполняем NaN
            train_df_clean = train_df[feature_columns].fillna(train_df[feature_columns].median())
            
            high_corr_pairs = find_highly_correlated_pairs(
                train_df_clean, 
                feature_columns, 
                args.correlation_threshold
            )
            
            if high_corr_pairs:
                correlated_features = select_features_to_remove(high_corr_pairs, feature_columns)
            else:
                correlated_features = set()
        
        features_by_reason['high_correlation'] = list(correlated_features)
        logger.info(f"   ✓ Найдено {len(correlated_features)} высококоррелированных фичей")
        
    except (NotImplementedError, ImportError) as e:
        logger.warning(f"   ⚠️  Анализ корреляции недоступен: {e}")
        features_by_reason['high_correlation'] = []
    except Exception as e:
        logger.warning(f"   ⚠️  Ошибка при анализе корреляции: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        features_by_reason['high_correlation'] = []
    
    # 3. Комплексный анализ (если не только корреляция)
    if not args.only_correlation:
        logger.info("\n4. Комплексный анализ фичей...")
        
        # Базовая статистика
        logger.info("   4.1. Вычисление базовой статистики...")
        stats_df = None
        try:
            stats_df = compute_basic_statistics(combined_df, all_features)
            
            # Фичи с 100% нулей
            logger.info("   4.2. Поиск фичей с 100% нулей...")
            zero_features = find_zero_features(stats_df)
            features_by_reason['all_zeros'] = zero_features
            logger.info(f"      ✓ Найдено {len(zero_features)} фичей с 100% нулей")
            
            # Фичи с большим процентом пропусков
            logger.info(f"   4.3. Поиск фичей с >{args.missing_threshold}% пропусков...")
            missing_features = find_high_missing_features(stats_df, args.missing_threshold)
            features_by_reason['high_missing'] = missing_features
            logger.info(f"      ✓ Найдено {len(missing_features)} фичей с большим процентом пропусков")
            
        except (NotImplementedError, ImportError) as e:
            logger.warning(f"   ⚠️  Комплексный анализ недоступен: {e}")
            features_by_reason['all_zeros'] = []
            features_by_reason['high_missing'] = []
        except Exception as e:
            logger.warning(f"   ⚠️  Ошибка при вычислении статистики: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            features_by_reason['all_zeros'] = []
            features_by_reason['high_missing'] = []
        
        # Анализ важности (если не отключен)
        if not args.no_low_importance and args.low_importance_percentile > 0:
            logger.info("   4.4. Анализ важности фичей...")
            try:
                if args.target in train_df.columns:
                    importance_df = analyze_feature_importance(
                        train_df, 
                        all_features, 
                        args.target
                    )
                    
                    low_importance = find_low_importance_features(
                        importance_df,
                        args.low_importance_percentile
                    )
                    features_by_reason['low_importance'] = low_importance
                    logger.info(f"      ✓ Найдено {len(low_importance)} фичей с низкой важностью")
                else:
                    logger.warning(f"      ⚠️  Целевая переменная '{args.target}' не найдена, пропускаем анализ важности")
                    features_by_reason['low_importance'] = []
            except (NotImplementedError, ImportError) as e:
                logger.warning(f"      ⚠️  Анализ важности недоступен: {e}")
                features_by_reason['low_importance'] = []
            except Exception as e:
                logger.warning(f"      ⚠️  Ошибка при анализе важности: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                features_by_reason['low_importance'] = []
        else:
            features_by_reason['low_importance'] = []
    
    # 5. Сохранение списка исключений
    logger.info("\n5. Формирование и сохранение списка исключений...")
    
    output_path = Path(args.output)
    save_excluded_features_grouped(
        features_by_reason,
        exclusions_file=output_path,
        correlation_threshold=args.correlation_threshold,
        missing_threshold=args.missing_threshold,
        low_importance_percentile=args.low_importance_percentile if not args.no_low_importance else 0
    )
    
    # Итоговая статистика
    logger.info("\n" + "=" * 80)
    logger.info("ИТОГОВАЯ СТАТИСТИКА")
    logger.info("=" * 80)
    
    total_excluded = sum(len(features) for features in features_by_reason.values())
    logger.info(f"Всего фичей для исключения: {total_excluded}")
    
    for reason, features in features_by_reason.items():
        if features:
            logger.info(f"  {reason}: {len(features)} фичей")
    
    logger.info(f"\n✓ Список сохранен в: {output_path}")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

