"""
Комплексный анализ фичей после удаления корреляций
Создает детальный отчет о всех фичах для понимания их характеристик перед обучением
"""
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, f_classif
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
    
    for col in feature_columns:
        if col not in df.columns:
            continue
            
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        # Проверяем, является ли колонка числовой
        is_numeric = pd.api.types.is_numeric_dtype(col_data)
        
        # Вычисляем статистику для бесконечных значений только для числовых типов
        if is_numeric:
            try:
                infinite_count = np.isinf(col_data).sum()
            except (TypeError, ValueError):
                infinite_count = 0
        else:
            infinite_count = 0
        
        stats_dict = {
            'feature': col,
            'count': len(col_data),
            'missing': df[col].isna().sum(),
            'missing_pct': (df[col].isna().sum() / len(df)) * 100,
            'zeros': (col_data == 0).sum() if is_numeric else 0,
            'zeros_pct': ((col_data == 0).sum() / len(col_data)) * 100 if (len(col_data) > 0 and is_numeric) else 0,
            'infinite': infinite_count,
            'infinite_pct': (infinite_count / len(col_data)) * 100 if len(col_data) > 0 else 0,
        }
        
        if is_numeric:
            stats_dict.update({
                'mean': col_data.mean(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max(),
                'median': col_data.median(),
                'q25': col_data.quantile(0.25),
                'q75': col_data.quantile(0.75),
                'skewness': col_data.skew(),
                'kurtosis': col_data.kurtosis(),
            })
            
            # Проверка нормальности (Shapiro-Wilk для небольших выборок, иначе используем только skew/kurtosis)
            if len(col_data) <= 5000:
                try:
                    _, p_value = stats.shapiro(col_data.sample(min(5000, len(col_data))))
                    stats_dict['normality_p_value'] = p_value
                    stats_dict['is_normal'] = p_value > 0.05
                except:
                    stats_dict['normality_p_value'] = None
                    stats_dict['is_normal'] = None
            else:
                stats_dict['normality_p_value'] = None
                stats_dict['is_normal'] = None
        else:
            stats_dict.update({
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'median': None,
                'q25': None,
                'q75': None,
                'skewness': None,
                'kurtosis': None,
                'normality_p_value': None,
                'is_normal': None,
            })
        
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
    try:
        mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
    except:
        mi_scores = np.zeros(len(feature_columns))
    
    # ANOVA F-score
    try:
        f_scores, f_pvalues = f_classif(X, y)
    except:
        f_scores = np.zeros(len(feature_columns))
        f_pvalues = np.ones(len(feature_columns))
    
    # Корреляция с таргетом (для числовых фичей)
    for i, col in enumerate(feature_columns):
        if col not in df.columns:
            continue
        
        corr_with_target = None
        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target_column]):
            try:
                corr = df[[col, target_column]].corr().iloc[0, 1]
                corr_with_target = corr if not np.isnan(corr) else None
            except:
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
        z_scores = np.abs(stats.zscore(col_data))
        zscore_outliers = (z_scores > 3).sum()
        zscore_outliers_pct = (zscore_outliers / len(col_data)) * 100
        
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

def generate_excluded_features_list(stats_df: pd.DataFrame,
                                    importance_df: pd.DataFrame,
                                    outliers_df: pd.DataFrame,
                                    all_features: List[str]) -> List[str]:
    """
    Генерирует список фичей для исключения на основе анализа
    
    Критерии исключения:
    1. Data leakage фичи (future_return_*, max_future_return, direction_*, future_volatility_*)
    2. Фичи с 100% нулей
    3. Фичи с >90% пропусков
    
    Args:
        stats_df: DataFrame со статистикой фичей
        importance_df: DataFrame с важностью фичей
        outliers_df: DataFrame с анализом выбросов
        all_features: Список всех фичей
    
    Returns:
        Список фичей для исключения
    """
    excluded = []
    
    # 1. Data leakage фичи (по паттернам)
    data_leakage_patterns = [
        'future_return', 'max_future_return', 
        'direction_', 'future_volatility_'
    ]
    for feature in all_features:
        for pattern in data_leakage_patterns:
            if pattern in feature.lower():
                excluded.append(feature)
                break
    
    # 2. Фичи с 100% нулей (из stats_df)
    if 'zeros' in stats_df.columns and 'count' in stats_df.columns:
        for _, row in stats_df.iterrows():
            feature = row['feature']
            if feature in excluded:
                continue
            
            count = row.get('count', 0)
            zeros = row.get('zeros', 0)
            zeros_pct = row.get('zeros_pct', 0)
            
            # Проверяем 100% нулей (либо zeros == count, либо zeros_pct == 100)
            if count > 0 and (zeros == count or zeros_pct >= 99.99):
                excluded.append(feature)
    
    # 3. Фичи с >90% пропусков
    if 'missing_pct' in stats_df.columns:
        for _, row in stats_df.iterrows():
            feature = row['feature']
            if feature in excluded:
                continue
            
            missing_pct = row.get('missing_pct', 0)
            if pd.notna(missing_pct) and missing_pct > 90.0:
                excluded.append(feature)
    
    # Убираем дубликаты и сортируем
    excluded = sorted(list(set(excluded)))
    
    return excluded

def create_html_report(stats_df: pd.DataFrame, importance_df: pd.DataFrame,
                      outliers_df: pd.DataFrame, class_stats_df: pd.DataFrame,
                      output_path: Path):
    """
    Создает HTML отчет с сводной информацией
    
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
            Всего фичей: {len(stats_df)}<br>
            Числовых фичей: {len(stats_df[stats_df['mean'].notna()])}<br>
            Фичей с пропусками: {len(stats_df[stats_df['missing'] > 0])}<br>
            Фичей с выбросами (>5%): {len(outliers_df[outliers_df['iqr_outliers_pct'] > 5])}
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
    
    top_features = importance_df.head(20)
    for _, row in top_features.iterrows():
        corr_str = f"{row['correlation_with_target']:.4f}" if pd.notna(row['correlation_with_target']) else "N/A"
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
    
    print(f"✓ HTML отчет сохранен: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Комплексный анализ фичей после удаления корреляций',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Полный анализ всех датасетов
  python analyze_features_comprehensive.py --train workspace/prepared/features/gold_train.csv --val workspace/prepared/features/gold_val.csv --test workspace/prepared/features/gold_test.csv
  
  # Только train датасет
  python analyze_features_comprehensive.py --train workspace/prepared/features/gold_train.csv
  
  # С генерацией графиков
  python analyze_features_comprehensive.py --train workspace/prepared/features/gold_train.csv --generate-plots
        """
    )
    
    parser.add_argument('--train', type=str, required=True,
                       help='Путь к train CSV файлу')
    parser.add_argument('--val', type=str, default=None,
                       help='Путь к validation CSV файлу (опционально)')
    parser.add_argument('--test', type=str, default=None,
                       help='Путь к test CSV файлу (опционально)')
    parser.add_argument('--target', type=str, default='signal_class',
                       help='Название целевой переменной (по умолчанию: signal_class)')
    parser.add_argument('--output-dir', type=str, default='workspace/features-analysis',
                       help='Директория для сохранения результатов (по умолчанию: workspace/features-analysis)')
    parser.add_argument('--top-features', type=int, default=50,
                       help='Количество топ фичей для детального анализа (по умолчанию: 50)')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Генерировать графики распределений и по классам')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("КОМПЛЕКСНЫЙ АНАЛИЗ ФИЧЕЙ")
    print("=" * 80)
    
    # Создаем выходную директорию
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные
    print(f"\n1. Загрузка данных...")
    train_df = pd.read_csv(args.train, index_col=0, parse_dates=True)
    print(f"   Train: {len(train_df)} образцов, {len(train_df.columns)} колонок")
    
    val_df = None
    if args.val:
        val_df = pd.read_csv(args.val, index_col=0, parse_dates=True)
        print(f"   Val: {len(val_df)} образцов, {len(val_df.columns)} колонок")
    
    test_df = None
    if args.test:
        test_df = pd.read_csv(args.test, index_col=0, parse_dates=True)
        print(f"   Test: {len(test_df)} образцов, {len(test_df.columns)} колонок")
    
    # Объединяем все датасеты для анализа
    combined_df = train_df.copy()
    if val_df is not None:
        combined_df = pd.concat([combined_df, val_df])
    if test_df is not None:
        combined_df = pd.concat([combined_df, test_df])
    
    print(f"   Объединенный датасет: {len(combined_df)} образцов")
    
    # Определяем фичи (исключаем таргет и временные колонки)
    feature_columns = [col for col in combined_df.columns 
                      if col != args.target and col != 'time']
    
    print(f"   Всего фичей для анализа: {len(feature_columns)}")
    
    # 1. Базовая статистика
    print(f"\n2. Вычисление базовой статистики...")
    stats_df = compute_basic_statistics(combined_df, feature_columns)
    stats_path = output_dir / 'feature_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"   ✓ Сохранено: {stats_path}")
    
    # 2. Анализ важности фичей
    print(f"\n3. Анализ важности фичей...")
    importance_df = analyze_feature_importance(train_df, feature_columns, args.target)
    importance_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✓ Сохранено: {importance_path}")
    print(f"   Топ-5 важных фичей:")
    for i, row in importance_df.head(5).iterrows():
        print(f"     {int(row['rank'])}. {row['feature']} (score: {row['combined_score']:.2f})")
    
    # 3. Анализ выбросов
    print(f"\n4. Анализ выбросов...")
    outliers_df = analyze_outliers(combined_df, feature_columns)
    outliers_path = output_dir / 'outliers_analysis.csv'
    outliers_df.to_csv(outliers_path, index=False)
    print(f"   ✓ Сохранено: {outliers_path}")
    print(f"   Топ-5 фичей с выбросами:")
    for i, row in outliers_df.head(5).iterrows():
        print(f"     {row['feature']}: {row['iqr_outliers_pct']:.2f}% выбросов")
    
    # 4. Анализ по классам
    print(f"\n5. Анализ распределений по классам...")
    class_stats_df = analyze_by_class(train_df, feature_columns, args.target)
    class_stats_path = output_dir / 'feature_by_class_statistics.csv'
    class_stats_df.to_csv(class_stats_path, index=False)
    print(f"   ✓ Сохранено: {class_stats_path}")
    
    # 5. HTML отчет
    print(f"\n6. Создание HTML отчета...")
    html_path = output_dir / 'feature_analysis_report.html'
    create_html_report(stats_df, importance_df, outliers_df, class_stats_df, html_path)
    
    # 7. Графики (опционально)
    if args.generate_plots:
        print(f"\n7. Генерация графиков...")
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # Графики распределений для топ фичей
        top_features = importance_df.head(args.top_features)['feature'].tolist()
        distributions_dir = plots_dir / 'distributions'
        distributions_dir.mkdir(exist_ok=True)
        
        for feature in top_features[:20]:  # Ограничиваем первыми 20 для скорости
            if feature not in combined_df.columns:
                continue
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Гистограмма
            combined_df[feature].dropna().hist(bins=50, ax=axes[0], edgecolor='black')
            axes[0].set_title(f'Распределение: {feature}')
            axes[0].set_xlabel('Значение')
            axes[0].set_ylabel('Частота')
            
            # Box plot
            combined_df[feature].dropna().plot.box(ax=axes[1])
            axes[1].set_title(f'Box Plot: {feature}')
            axes[1].set_ylabel('Значение')
            
            plt.tight_layout()
            plt.savefig(distributions_dir / f'{feature}_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   ✓ Графики распределений сохранены в {distributions_dir}")
        
        # Графики по классам для топ фичей
        by_class_dir = plots_dir / 'by_class'
        by_class_dir.mkdir(exist_ok=True)
        
        for feature in top_features[:20]:
            if feature not in train_df.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Box plot по классам
            class_data = []
            class_labels = []
            for class_val in sorted(train_df[args.target].unique()):
                class_values = train_df[train_df[args.target] == class_val][feature].dropna()
                if len(class_values) > 0:
                    class_data.append(class_values)
                    class_labels.append(f'Class {class_val}')
            
            if class_data:
                ax.boxplot(class_data, labels=class_labels)
                ax.set_title(f'Распределение по классам: {feature}')
                ax.set_ylabel('Значение')
                ax.set_xlabel('Класс')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(by_class_dir / f'{feature}_by_class.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        print(f"   ✓ Графики по классам сохранены в {by_class_dir}")
    
    # 8. Создание списка фичей для исключения
    print(f"\n8. Создание списка фичей для исключения...")
    excluded_features = generate_excluded_features_list(
        stats_df=stats_df,
        importance_df=importance_df,
        outliers_df=outliers_df,
        all_features=feature_columns
    )
    
    if excluded_features:
        from utils.feature_exclusions import save_excluded_features
        exclusions_file = output_dir / 'excluded_features.txt'
        save_excluded_features(excluded_features, exclusions_file)
        print(f"   ✓ Сохранено {len(excluded_features)} фичей для исключения: {exclusions_file}")
        print(f"   Топ-10 исключаемых фичей:")
        for i, feat in enumerate(excluded_features[:10], 1):
            print(f"     {i}. {feat}")
        if len(excluded_features) > 10:
            print(f"     ... и ещё {len(excluded_features) - 10} фичей")
    else:
        print(f"   ℹ Не найдено фичей для автоматического исключения")
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)
    print(f"\nРезультаты сохранены в: {output_dir}")
    print(f"  - feature_statistics.csv - базовая статистика")
    print(f"  - feature_importance.csv - важность фичей")
    print(f"  - outliers_analysis.csv - анализ выбросов")
    print(f"  - feature_by_class_statistics.csv - статистика по классам")
    print(f"  - feature_analysis_report.html - сводный HTML отчет")
    if excluded_features:
        print(f"  - excluded_features.txt - список фичей для исключения ({len(excluded_features)} фичей)")
    if args.generate_plots:
        print(f"  - plots/ - графики распределений и по классам")

if __name__ == '__main__':
    main()

