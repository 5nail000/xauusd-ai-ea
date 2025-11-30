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

# Дополнительные импорты для комплексного анализа
try:
    from scipy import stats
except ImportError:
    stats = None

try:
    from sklearn.feature_selection import mutual_info_classif, f_classif
except ImportError:
    mutual_info_classif = None
    f_classif = None

# Настройка логирования (должно быть до импортов, которые могут использовать logger)
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Защищенные фичи - никогда не удаляются
PROTECTED_FEATURES = ['open', 'high', 'low', 'close']

# Константы
SHAPIRO_WILK_MAX_SAMPLE = 5000
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_MISSING_THRESHOLD = 90.0


# ============================================================================
# ФУНКЦИИ ИЗ analyze_feature_correlation.py
# ============================================================================

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
    
    # Используем numpy для эффективного поиска верхнего треугольника
    # (исключаем диагональ и дубликаты)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    high_corr = corr_matrix.where(mask).stack()
    
    # Фильтруем по порогу
    high_corr = high_corr[high_corr.abs() > threshold]
    
    # Преобразуем в список кортежей
    high_corr_pairs = [
        (idx[0], idx[1], float(val))
        for idx, val in high_corr.items()
    ]
    
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


def analyze_combined_datasets(train_path: str, val_path: str, test_path: str,
                              threshold: float = 0.95, return_pairs: bool = False):
    """
    Анализирует корреляции на объединенном датасете (train+val+test)
    и возвращает список фичей для удаления
    
    Args:
        train_path: Путь к train CSV
        val_path: Путь к val CSV
        test_path: Путь к test CSV
        threshold: Порог корреляции
        return_pairs: Если True, возвращает также список корреляционных пар
    
    Returns:
        Множество фичей для удаления (или кортеж (features_to_remove, high_corr_pairs) если return_pairs=True)
    """
    logger.info("=" * 80)
    logger.info("АНАЛИЗ КОРРЕЛЯЦИИ НА ОБЪЕДИНЕННОМ ДАТАСЕТЕ")
    logger.info("=" * 80)
    
    # Загружаем все три датасета
    datasets = {}
    for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not Path(path).exists():
            logger.warning(f"⚠️  Файл {path} не найден, пропускаем...")
            continue
        logger.info(f"\nЗагрузка {name} данных из {path}...")
        datasets[name] = pd.read_csv(path, index_col=0, parse_dates=True)
        logger.info(f"   Загружено {len(datasets[name])} строк, {len(datasets[name].columns)} колонок")
    
    if not datasets:
        logger.error("❌ Ошибка: Не найдено ни одного датасета для анализа")
        if return_pairs:
            return set(), []
        return set()
    
    # Объединяем все датасеты
    logger.info("\nОбъединение датасетов для анализа...")
    combined_df = pd.concat(datasets.values(), ignore_index=False)
    logger.info(f"   Объединенный датасет: {len(combined_df)} строк, {len(combined_df.columns)} колонок")
    
    # Выбор фичей (исключаем целевые переменные)
    exclude_patterns_set = {'future_return', 'signal_class', 'signal_class_name', 'max_future_return'}
    feature_columns = [
        col for col in combined_df.columns 
        if col not in exclude_patterns_set
        and pd.api.types.is_numeric_dtype(combined_df[col])
    ]
    
    logger.info(f"   Найдено {len(feature_columns)} фичей для анализа")
    
    # Проверка на NaN
    logger.info("\nПроверка данных...")
    nan_counts = combined_df[feature_columns].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    
    # Исключаем фичи с >50% пропусков из корреляционного анализа
    high_missing_threshold = len(combined_df) * 0.5
    high_missing_cols = nan_counts[nan_counts > high_missing_threshold].index.tolist()
    if high_missing_cols:
        logger.info(f"   ⚠️  {len(high_missing_cols)} фичей с >50% пропусков исключены из корреляционного анализа")
        feature_columns = [col for col in feature_columns if col not in high_missing_cols]
    
    if len(cols_with_nan) > 0:
        logger.info(f"   ⚠️  Найдено {len(cols_with_nan)} фичей с NaN значениями")
        logger.info(f"   Заполняем NaN медианой...")
        combined_df[feature_columns] = combined_df[feature_columns].fillna(combined_df[feature_columns].median())
    else:
        logger.info("   ✓ NaN значений не найдено")
    
    # Анализ корреляции на объединенном датасете
    logger.info(f"\nАнализ корреляции на объединенном датасете (порог: {threshold})...")
    high_corr_pairs = find_highly_correlated_pairs(combined_df, feature_columns, threshold)
    
    if len(high_corr_pairs) == 0:
        logger.info(f"   ✓ Высококоррелированных пар (>{threshold}) не найдено")
        if return_pairs:
            return set(), []
        return set()
    
    logger.info(f"   Найдено {len(high_corr_pairs)} высококоррелированных пар")
    
    # Выбор фичей для удаления
    logger.info("\nВыбор фичей для удаления...")
    logger.info(f"   Защищенные фичи (никогда не удаляются): {', '.join(PROTECTED_FEATURES)}")
    features_to_remove = select_features_to_remove(high_corr_pairs, feature_columns)
    
    logger.info(f"\n✓ Будет удалено {len(features_to_remove)} фичей из всех датасетов")
    for feat in sorted(list(features_to_remove)[:10]):
        logger.info(f"     - {feat}")
    if len(features_to_remove) > 10:
        logger.info(f"     ... и еще {len(features_to_remove) - 10} фичей")
    
    if return_pairs:
        return features_to_remove, high_corr_pairs
    return features_to_remove


# ============================================================================
# ФУНКЦИИ ИЗ analyze_features_comprehensive.py
# ============================================================================

def compute_basic_statistics(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Вычисляет базовую статистику по фичам
    
    Args:
        df: DataFrame с данными
        feature_columns: Список колонок-фичей
    
    Returns:
        DataFrame со статистикой
    """
    stats_list = []
    
    # Фильтруем только существующие числовые фичи
    numeric_features = [
        col for col in feature_columns 
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    total_rows = len(df)
    
    for col in numeric_features:
        col_data = df[col]
        non_null = col_data.dropna()
        
        if len(non_null) == 0:
            continue
        
        # Вычисляем missing один раз
        missing_count = col_data.isna().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        
        # Вычисляем бесконечные значения
        try:
            infinite_count = np.isinf(non_null).sum()
        except (TypeError, ValueError):
            infinite_count = 0
        
        # Вычисляем нули
        zeros_count = (non_null == 0).sum()
        zeros_pct = (zeros_count / len(non_null)) * 100 if len(non_null) > 0 else 0
        
        stats_dict = {
            'feature': col,
            'count': len(non_null),
            'missing': missing_count,
            'missing_pct': missing_pct,
            'zeros': zeros_count,
            'zeros_pct': zeros_pct,
            'infinite': infinite_count,
            'infinite_pct': (infinite_count / len(non_null)) * 100 if len(non_null) > 0 else 0,
            'mean': non_null.mean(),
            'std': non_null.std(),
            'min': non_null.min(),
            'max': non_null.max(),
            'median': non_null.median(),
            'q25': non_null.quantile(0.25),
            'q75': non_null.quantile(0.75),
            'skewness': non_null.skew(),
            'kurtosis': non_null.kurtosis(),
        }
        
        # Проверка нормальности (Shapiro-Wilk для небольших выборок)
        if len(non_null) <= SHAPIRO_WILK_MAX_SAMPLE and stats is not None:
            try:
                sample_size = min(SHAPIRO_WILK_MAX_SAMPLE, len(non_null))
                _, p_value = stats.shapiro(non_null.sample(sample_size, random_state=42))
                stats_dict['normality_p_value'] = p_value
                stats_dict['is_normal'] = p_value > 0.05
            except (ValueError, TypeError, RuntimeError) as e:
                logger.debug(f"Ошибка при проверке нормальности для {col}: {e}")
                stats_dict['normality_p_value'] = None
                stats_dict['is_normal'] = None
        else:
            stats_dict['normality_p_value'] = None
            stats_dict['is_normal'] = None
        
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)


def analyze_feature_importance(df: pd.DataFrame, feature_columns: List[str], 
                               target_column: str) -> pd.DataFrame:
    """
    Анализирует важность фичей через Mutual Information и корреляцию с таргетом
    
    Args:
        df: DataFrame с данными
        feature_columns: Список колонок-фичей
        target_column: Название целевой переменной
    
    Returns:
        DataFrame с важностью фичей
    """
    importance_list = []
    
    # Подготовка данных
    X = df[feature_columns].fillna(0).replace([np.inf, -np.inf], 0)
    y = df[target_column].values
    
    # Mutual Information
    mi_scores = np.zeros(len(feature_columns))
    if mutual_info_classif is not None:
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при вычислении Mutual Information: {e}")
            pass
    
    # ANOVA F-score
    f_scores = np.zeros(len(feature_columns))
    f_pvalues = np.ones(len(feature_columns))
    if f_classif is not None:
        try:
            f_scores, f_pvalues = f_classif(X, y)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.debug(f"Ошибка при вычислении F-score: {e}")
            pass
    
    # Корреляция с таргетом (для числовых фичей)
    for i, col in enumerate(feature_columns):
        if col not in df.columns:
            continue
        
        corr_with_target = None
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_column]):
            try:
                corr = df[[col, target_column]].corr().iloc[0, 1]
                corr_with_target = corr if not np.isnan(corr) else None
            except (ValueError, TypeError, KeyError, IndexError) as e:
                logger.debug(f"Ошибка при вычислении корреляции с таргетом для {col}: {e}")
                pass
        
        importance_list.append({
            'feature': col,
            'mutual_info': mi_scores[i] if i < len(mi_scores) else 0,
            'f_score': f_scores[i] if i < len(f_scores) else 0,
            'f_pvalue': f_pvalues[i] if i < len(f_pvalues) else 1,
            'correlation_with_target': corr_with_target,
        })
    
    importance_df = pd.DataFrame(importance_list)
    
    # Нормализуем scores для ранжирования
    if importance_df['mutual_info'].max() > 0:
        importance_df['mi_normalized'] = (importance_df['mutual_info'] / importance_df['mutual_info'].max()) * 100
    else:
        importance_df['mi_normalized'] = 0
    
    if importance_df['f_score'].max() > 0:
        importance_df['f_normalized'] = (importance_df['f_score'] / importance_df['f_score'].max()) * 100
    else:
        importance_df['f_normalized'] = 0
    
    # Комбинированный score (среднее нормализованных значений)
    importance_df['combined_score'] = (
        importance_df['mi_normalized'] * 0.5 + 
        importance_df['f_normalized'] * 0.5
    )
    
    # Ранжирование
    importance_df = importance_df.sort_values('combined_score', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


def analyze_outliers(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Анализирует выбросы в фичах
    
    Args:
        df: DataFrame с данными
        feature_columns: Список колонок-фичей
    
    Returns:
        DataFrame с анализом выбросов
    """
    outliers_list = []
    
    for col in feature_columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if len(col_data) == 0 or not pd.api.types.is_numeric_dtype(col_data):
            continue
        
        # IQR метод
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            iqr_outliers_pct = (iqr_outliers / len(col_data)) * 100
        else:
            iqr_outliers = 0
            iqr_outliers_pct = 0
        
        # Z-score метод (|z| > 3)
        zscore_outliers = 0
        zscore_outliers_pct = 0
        if stats is not None:
            try:
                z_scores = np.abs(stats.zscore(col_data))
                zscore_outliers = (z_scores > 3).sum()
                zscore_outliers_pct = (zscore_outliers / len(col_data)) * 100
            except (ValueError, TypeError, RuntimeError) as e:
                logger.debug(f"Ошибка при вычислении Z-score для {col}: {e}")
                pass
        
        outliers_list.append({
            'feature': col,
            'iqr_outliers': iqr_outliers,
            'iqr_outliers_pct': iqr_outliers_pct,
            'zscore_outliers': zscore_outliers,
            'zscore_outliers_pct': zscore_outliers_pct,
            'total_outliers_iqr': iqr_outliers,
            'total_outliers_zscore': zscore_outliers,
        })
    
    outliers_df = pd.DataFrame(outliers_list)
    if len(outliers_df) > 0:
        outliers_df = outliers_df.sort_values('iqr_outliers_pct', ascending=False)
    
    return outliers_df


def analyze_by_class(df: pd.DataFrame, feature_columns: List[str], 
                     target_column: str) -> pd.DataFrame:
    """
    Анализирует распределение фичей по классам
    
    Args:
        df: DataFrame с данными
        feature_columns: Список колонок-фичей
        target_column: Название целевой переменной
    
    Returns:
        DataFrame со статистикой по классам
    """
    class_stats = []
    
    unique_classes = sorted(df[target_column].unique())
    
    for col in feature_columns:
        if col not in df.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        for class_val in unique_classes:
            class_data = df[df[target_column] == class_val][col].dropna()
            
            if len(class_data) == 0:
                continue
            
            class_stats.append({
                'feature': col,
                'class': class_val,
                'count': len(class_data),
                'mean': class_data.mean(),
                'std': class_data.std(),
                'median': class_data.median(),
                'min': class_data.min(),
                'max': class_data.max(),
            })
        
        # Вычисляем дифференциацию между классами (коэффициент вариации между классами)
        class_means = []
        for class_val in unique_classes:
            class_data = df[df[target_column] == class_val][col].dropna()
            if len(class_data) > 0:
                class_means.append(class_data.mean())
        
        if len(class_means) > 1:
            cv_between_classes = np.std(class_means) / (np.mean(class_means) + 1e-10)
        else:
            cv_between_classes = 0
        
        # Добавляем общую статистику дифференциации
        class_stats.append({
            'feature': col,
            'class': 'differentiation_cv',
            'count': len(df),
            'mean': cv_between_classes,
            'std': None,
            'median': None,
            'min': None,
            'max': None,
        })
    
    return pd.DataFrame(class_stats)


def create_html_report(stats_df: pd.DataFrame, importance_df: pd.DataFrame,
                      outliers_df: pd.DataFrame, class_stats_df: pd.DataFrame,
                      output_path: Path):
    """
    Создает HTML отчет со сводной информацией
    
    Args:
        stats_df: DataFrame со статистикой
        importance_df: DataFrame с важностью фичей
        outliers_df: DataFrame с анализом выбросов
        class_stats_df: DataFrame со статистикой по классам
        output_path: Путь для сохранения отчета
    """
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Комплексный анализ фичей</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #4CAF50; color: white; font-weight: bold; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f5e9; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        .warning {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; margin: 10px 0; }}
        .info {{ background-color: #d1ecf1; padding: 10px; border-left: 4px solid #0c5460; margin: 10px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Комплексный анализ фичей</h1>
        
        <div class="info">
            <strong>Общая информация:</strong><br>
            Всего фичей: {len(stats_df) if stats_df is not None else 0}<br>
            Числовых фичей: {len(stats_df[stats_df['mean'].notna()]) if stats_df is not None and 'mean' in stats_df.columns else 0}<br>
            Фичей с пропусками: {len(stats_df[stats_df['missing'] > 0]) if stats_df is not None and 'missing' in stats_df.columns else 0}<br>
            Фичей с выбросами (>5%): {len(outliers_df[outliers_df['iqr_outliers_pct'] > 5]) if outliers_df is not None and len(outliers_df) > 0 and 'iqr_outliers_pct' in outliers_df.columns else 0}
        </div>
        
        <h2>Топ-20 важных фичей</h2>
        <table>
            <tr>
                <th>Ранг</th>
                <th>Фича</th>
                <th>Mutual Info</th>
                <th>F-Score</th>
                <th>Комбинированный Score</th>
                <th>Корреляция с таргетом</th>
            </tr>
"""
    
    if importance_df is not None and len(importance_df) > 0:
        top_features = importance_df.head(20)
        for _, row in top_features.iterrows():
            corr_str = f"{row['correlation_with_target']:.4f}" if pd.notna(row.get('correlation_with_target')) else "N/A"
            html_content += f"""
            <tr>
                <td>{int(row['rank'])}</td>
                <td>{row['feature']}</td>
                <td>{row['mutual_info']:.4f}</td>
                <td>{row['f_score']:.2f}</td>
                <td>{row['combined_score']:.2f}</td>
                <td>{corr_str}</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <h2>Топ-10 фичей с наибольшим количеством выбросов</h2>
        <table>
            <tr>
                <th>Фича</th>
                <th>Выбросы (IQR)</th>
                <th>Выбросы (%)</th>
                <th>Выбросы (Z-score)</th>
            </tr>
"""
    
    if outliers_df is not None and len(outliers_df) > 0:
        top_outliers = outliers_df.head(10)
        for _, row in top_outliers.iterrows():
            html_content += f"""
            <tr>
                <td>{row['feature']}</td>
                <td>{int(row['iqr_outliers'])}</td>
                <td>{row['iqr_outliers_pct']:.2f}%</td>
                <td>{int(row['zscore_outliers'])}</td>
            </tr>
"""
    
    html_content += """
        </table>
        
        <div class="warning">
            <strong>Рекомендации:</strong><br>
            • Фичи с высоким процентом выбросов (>10%) могут требовать обработки или удаления<br>
            • Фичи с низкой важностью (< 1% комбинированного score) можно рассмотреть для удаления<br>
            • Фичи с высокой корреляцией с таргетом (>0.3 или <-0.3) особенно важны для модели
        </div>
    </div>
</body>
</html>
"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"✓ HTML отчет сохранен: {output_path}")


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


def load_and_combine_datasets(train_path: Path, val_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Загружает и объединяет датасеты
    
    Args:
        train_path: Путь к train CSV
        val_path: Путь к val CSV
        test_path: Путь к test CSV
    
    Returns:
        Кортеж (train_df, val_df, test_df, combined_df)
    """
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
    
    # Объединяем для анализа (одно concat вместо множественных)
    dfs_to_concat = [train_df]
    if val_df is not None:
        dfs_to_concat.append(val_df)
    if test_df is not None:
        dfs_to_concat.append(test_df)
    combined_df = pd.concat(dfs_to_concat, ignore_index=False)
    
    logger.info(f"   Объединенный датасет: {len(combined_df)} образцов")
    
    return train_df, val_df, test_df, combined_df


def get_feature_columns(combined_df: pd.DataFrame) -> List[str]:
    """
    Определяет список фичей для анализа (исключая целевые переменные)
    
    Args:
        combined_df: Объединенный DataFrame
    
    Returns:
        Список фичей для анализа
    """
    exclude_patterns_set = {'future_return', 'signal_class', 'signal_class_name', 'max_future_return'}
    all_features = [
        col for col in combined_df.columns 
        if col not in exclude_patterns_set
        and pd.api.types.is_numeric_dtype(combined_df[col])
    ]
    return all_features


def perform_correlation_analysis(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                                train_path: Path, val_path: Path, test_path: Path,
                                threshold: float, save_details: bool) -> Tuple[Set[str], List[Tuple[str, str, float]]]:
    """
    Выполняет анализ корреляции
    
    Args:
        train_df: Train DataFrame
        val_df: Val DataFrame (может быть None)
        test_df: Test DataFrame (может быть None)
        train_path: Путь к train CSV
        val_path: Путь к val CSV
        test_path: Путь к test CSV
        threshold: Порог корреляции
        save_details: Сохранять ли детальные результаты
    
    Returns:
        Кортеж (correlated_features, high_corr_pairs)
    """
    logger.info(f"\n3. Анализ корреляции (порог: {threshold})...")
    high_corr_pairs = []
    correlated_features = set()
    
    try:
        if val_path.exists() and test_path.exists():
            # Анализ на объединенном датасете
            if save_details:
                result = analyze_combined_datasets(
                    str(train_path),
                    str(val_path),
                    str(test_path),
                    threshold=threshold,
                    return_pairs=True
                )
                correlated_features, high_corr_pairs = result
            else:
                correlated_features = analyze_combined_datasets(
                    str(train_path),
                    str(val_path),
                    str(test_path),
                    threshold=threshold
                )
                high_corr_pairs = []
        else:
            # Анализ только на train
            logger.info("   ⚠️  Val/Test не найдены, анализируем только train")
            exclude_patterns_set = {'future_return', 'signal_class', 'signal_class_name', 'max_future_return'}
            feature_columns = [
                col for col in train_df.columns 
                if col not in exclude_patterns_set
                and pd.api.types.is_numeric_dtype(train_df[col])
            ]
            
            # Заполняем NaN
            train_df_clean = train_df[feature_columns].fillna(train_df[feature_columns].median())
            
            high_corr_pairs = find_highly_correlated_pairs(
                train_df_clean, 
                feature_columns, 
                threshold
            )
            
            if high_corr_pairs:
                correlated_features = select_features_to_remove(high_corr_pairs, feature_columns)
            else:
                correlated_features = set()
        
        logger.info(f"   ✓ Найдено {len(correlated_features)} высококоррелированных фичей")
        
    except (NotImplementedError, ImportError) as e:
        logger.warning(f"   ⚠️  Анализ корреляции недоступен: {e}")
        correlated_features = set()
        high_corr_pairs = []
    except Exception as e:
        logger.warning(f"   ⚠️  Ошибка при анализе корреляции: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        correlated_features = set()
        high_corr_pairs = []
    
    return correlated_features, high_corr_pairs


def perform_comprehensive_analysis(combined_df: pd.DataFrame, train_df: pd.DataFrame,
                                   all_features: List[str], target: str, args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Выполняет комплексный анализ фичей
    
    Args:
        combined_df: Объединенный DataFrame
        train_df: Train DataFrame
        all_features: Список фичей для анализа
        target: Название целевой переменной
        args: Аргументы командной строки
    
    Returns:
        Кортеж (stats_df, importance_df, outliers_df, class_stats_df)
    """
    logger.info("\n4. Комплексный анализ фичей...")
    
    stats_df = None
    importance_df = None
    outliers_df = None
    class_stats_df = None
    
    # Базовая статистика
    logger.info("   4.1. Вычисление базовой статистики...")
    try:
        stats_df = compute_basic_statistics(combined_df, all_features)
        
        # Фичи с 100% нулей
        logger.info("   4.2. Поиск фичей с 100% нулей...")
        zero_features = find_zero_features(stats_df)
        logger.info(f"      ✓ Найдено {len(zero_features)} фичей с 100% нулей")
        
        # Фичи с большим процентом пропусков
        logger.info(f"   4.3. Поиск фичей с >{args.missing_threshold}% пропусков...")
        missing_features = find_high_missing_features(stats_df, args.missing_threshold)
        logger.info(f"      ✓ Найдено {len(missing_features)} фичей с большим процентом пропусков")
        
        # Анализ выбросов (для детальных результатов)
        if args.save_details:
            logger.info("   4.4. Анализ выбросов...")
            try:
                outliers_df = analyze_outliers(combined_df, all_features)
                logger.info(f"      ✓ Проанализировано {len(outliers_df)} фичей на выбросы")
            except Exception as e:
                logger.warning(f"      ⚠️  Ошибка при анализе выбросов: {e}")
                outliers_df = None
        
    except (NotImplementedError, ImportError) as e:
        logger.warning(f"   ⚠️  Комплексный анализ недоступен: {e}")
    except Exception as e:
        logger.warning(f"   ⚠️  Ошибка при вычислении статистики: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    # Анализ важности (если не отключен)
    if not args.no_low_importance and args.low_importance_percentile > 0:
        logger.info("   4.5. Анализ важности фичей...")
        try:
            if target in train_df.columns:
                importance_df = analyze_feature_importance(
                    train_df, 
                    all_features, 
                    target
                )
                
                low_importance = find_low_importance_features(
                    importance_df,
                    args.low_importance_percentile
                )
                logger.info(f"      ✓ Найдено {len(low_importance)} фичей с низкой важностью")
            else:
                logger.warning(f"      ⚠️  Целевая переменная '{target}' не найдена, пропускаем анализ важности")
        except (NotImplementedError, ImportError) as e:
            logger.warning(f"      ⚠️  Анализ важности недоступен: {e}")
        except Exception as e:
            logger.warning(f"      ⚠️  Ошибка при анализе важности: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    # Анализ по классам (для детальных результатов)
    if args.save_details and target in train_df.columns:
        logger.info("   4.6. Анализ распределения фичей по классам...")
        try:
            class_stats_df = analyze_by_class(train_df, all_features, target)
            logger.info(f"      ✓ Проанализировано {len(class_stats_df)} комбинаций фича-класс")
        except Exception as e:
            logger.warning(f"      ⚠️  Ошибка при анализе по классам: {e}")
            class_stats_df = None
    
    return stats_df, importance_df, outliers_df, class_stats_df


def save_detailed_results(high_corr_pairs: List[Tuple[str, str, float]],
                         stats_df: pd.DataFrame, importance_df: pd.DataFrame,
                         outliers_df: pd.DataFrame, class_stats_df: pd.DataFrame,
                         correlation_threshold: float):
    """
    Сохраняет детальные результаты анализа
    
    Args:
        high_corr_pairs: Список корреляционных пар
        stats_df: DataFrame со статистикой
        importance_df: DataFrame с важностью фичей
        outliers_df: DataFrame с анализом выбросов
        class_stats_df: DataFrame со статистикой по классам
        correlation_threshold: Порог корреляции
    """
    logger.info("\n6. Сохранение детальных результатов анализа...")
    output_dir = Path('workspace/analysis-of-features')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохранение корреляционных пар
    if high_corr_pairs:
        logger.info("   6.1. Сохранение корреляционных пар...")
        pairs_df = pd.DataFrame(high_corr_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])
        pairs_df['Abs_Correlation'] = pairs_df['Correlation'].abs()
        pairs_df = pairs_df.sort_values('Abs_Correlation', ascending=False)
        pairs_path = output_dir / f'highly_correlated_pairs_threshold_{correlation_threshold:.2f}.csv'
        pairs_df.to_csv(pairs_path, index=False)
        logger.info(f"      ✓ Сохранено: {pairs_path}")
    
    # Сохранение статистики фичей
    if stats_df is not None:
        logger.info("   6.2. Сохранение базовой статистики...")
        stats_path = output_dir / 'feature_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"      ✓ Сохранено: {stats_path}")
    
    # Сохранение важности фичей
    if importance_df is not None:
        logger.info("   6.3. Сохранение важности фичей...")
        importance_path = output_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"      ✓ Сохранено: {importance_path}")
    
    # Сохранение анализа выбросов
    if outliers_df is not None:
        logger.info("   6.4. Сохранение анализа выбросов...")
        outliers_path = output_dir / 'outliers_analysis.csv'
        outliers_df.to_csv(outliers_path, index=False)
        logger.info(f"      ✓ Сохранено: {outliers_path}")
    
    # Сохранение статистики по классам
    if class_stats_df is not None:
        logger.info("   6.5. Сохранение статистики по классам...")
        class_stats_path = output_dir / 'feature_by_class_statistics.csv'
        class_stats_df.to_csv(class_stats_path, index=False)
        logger.info(f"      ✓ Сохранено: {class_stats_path}")
    
    # Создание HTML отчета
    logger.info("   6.6. Создание HTML отчета...")
    html_path = output_dir / 'feature_analysis_report.html'
    try:
        create_html_report(stats_df, importance_df, outliers_df, class_stats_df, html_path)
        logger.info(f"      ✓ Сохранено: {html_path}")
    except Exception as e:
        logger.warning(f"      ⚠️  Ошибка при создании HTML отчета: {e}")
    
    logger.info(f"\n   ✓ Все детальные результаты сохранены в: {output_dir}")


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

  # С сохранением детальных результатов (CSV файлы и HTML отчет)
  python analyze_and_exclude_features.py --save-details
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
    
    parser.add_argument(
        '--save-details',
        action='store_true',
        help='Сохранить детальные результаты анализа в workspace/analysis-of-features/ (CSV файлы и HTML отчет)'
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
    
    # Загружаем и объединяем данные
    train_df, val_df, test_df, combined_df = load_and_combine_datasets(train_path, val_path, test_path)
    
    # Определяем фичи
    all_features = get_feature_columns(combined_df)
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
    correlated_features, high_corr_pairs = perform_correlation_analysis(
        train_df, val_df, test_df,
        train_path, val_path, test_path,
        args.correlation_threshold, args.save_details
    )
    features_by_reason['high_correlation'] = list(correlated_features)
    
    # 3. Комплексный анализ (если не только корреляция)
    stats_df = None
    importance_df = None
    outliers_df = None
    class_stats_df = None
    
    if not args.only_correlation:
        stats_df, importance_df, outliers_df, class_stats_df = perform_comprehensive_analysis(
            combined_df, train_df, all_features, args.target, args
        )
        
        # Добавляем найденные фичи в features_by_reason
        if stats_df is not None:
            zero_features = find_zero_features(stats_df)
            features_by_reason['all_zeros'] = zero_features
            
            missing_features = find_high_missing_features(stats_df, args.missing_threshold)
            features_by_reason['high_missing'] = missing_features
        
        if importance_df is not None and not args.no_low_importance and args.low_importance_percentile > 0:
            low_importance = find_low_importance_features(
                importance_df,
                args.low_importance_percentile
            )
            features_by_reason['low_importance'] = low_importance
        else:
            features_by_reason['low_importance'] = []
    
    # 4. Сохранение списка исключений
    logger.info("\n5. Формирование и сохранение списка исключений...")
    output_path = Path(args.output)
    save_excluded_features_grouped(
        features_by_reason,
        exclusions_file=output_path,
        correlation_threshold=args.correlation_threshold,
        missing_threshold=args.missing_threshold,
        low_importance_percentile=args.low_importance_percentile if not args.no_low_importance else 0
    )
    
    # 5. Сохранение детальных результатов (если запрошено)
    if args.save_details:
        save_detailed_results(
            high_corr_pairs, stats_df, importance_df, outliers_df, class_stats_df,
            args.correlation_threshold
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
    
    logger.info(f"\n✓ Список исключений сохранен в: {output_path}")
    if args.save_details:
        logger.info(f"✓ Детальные результаты сохранены в: workspace/analysis-of-features/")
    logger.info("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

