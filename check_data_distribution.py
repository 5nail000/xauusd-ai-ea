"""
Скрипт для проверки распределения данных в train/val/test выборках
Помогает выявить проблемы с разделением данных и утечки данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Опциональные импорты
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy не установлен. Статистические тесты будут пропущены.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("matplotlib/seaborn не установлены. Визуализации будут пропущены.")

# Настройка для корректного отображения времени в логах
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_datasets():
    """Загружает train/val/test датасеты"""
    train_path = Path('workspace/prepared/features/gold_train.csv')
    val_path = Path('workspace/prepared/features/gold_val.csv')
    test_path = Path('workspace/prepared/features/gold_test.csv')
    
    if not train_path.exists():
        logger.error(f"Файл не найден: {train_path}")
        logger.error("Сначала запустите: python prepare_gold_data.py --months <N>")
        raise FileNotFoundError(f"Файл не найден: {train_path}")
    if not val_path.exists():
        logger.error(f"Файл не найден: {val_path}")
        logger.error("Сначала запустите: python prepare_gold_data.py --months <N>")
        raise FileNotFoundError(f"Файл не найден: {val_path}")
    if not test_path.exists():
        logger.error(f"Файл не найден: {test_path}")
        logger.error("Сначала запустите: python prepare_gold_data.py --months <N>")
        raise FileNotFoundError(f"Файл не найден: {test_path}")
    
    logger.info(f"Загрузка данных из {train_path.parent}")
    train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
    val_df = pd.read_csv(val_path, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_path, index_col=0, parse_dates=True)
    
    logger.info(f"  Train: {len(train_df)} образцов")
    logger.info(f"  Val:   {len(val_df)} образцов")
    logger.info(f"  Test:  {len(test_df)} образцов")
    
    return train_df, val_df, test_df

def check_temporal_distribution(train_df, val_df, test_df):
    """Проверяет временное распределение данных"""
    logger.info("\n" + "=" * 80)
    logger.info("ВРЕМЕННОЕ РАСПРЕДЕЛЕНИЕ")
    logger.info("=" * 80)
    
    train_start = train_df.index[0]
    train_end = train_df.index[-1]
    val_start = val_df.index[0]
    val_end = val_df.index[-1]
    test_start = test_df.index[0]
    test_end = test_df.index[-1]
    
    logger.info(f"Train:  {train_start} - {train_end} ({len(train_df)} образцов)")
    logger.info(f"Val:    {val_start} - {val_end} ({len(val_df)} образцов)")
    logger.info(f"Test:   {test_start} - {test_end} ({len(test_df)} образцов)")
    
    # Проверка на перекрытие
    issues = []
    if train_end >= val_start:
        issues.append(f"⚠️  ПЕРЕКРЫТИЕ: Train заканчивается ({train_end}) после начала Val ({val_start})")
    if val_end >= test_start:
        issues.append(f"⚠️  ПЕРЕКРЫТИЕ: Val заканчивается ({val_end}) после начала Test ({test_start})")
    if train_end >= test_start:
        issues.append(f"⚠️  ПЕРЕКРЫТИЕ: Train заканчивается ({train_end}) после начала Test ({test_start})")
    
    if issues:
        logger.warning("\nОбнаружены проблемы:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("✓ Временное разделение корректно (нет перекрытий)")
    
    return {
        'train_start': train_start,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
        'test_start': test_start,
        'test_end': test_end,
        'has_overlap': len(issues) > 0
    }

def check_class_distribution(train_df, val_df, test_df, target_column='signal_class'):
    """Проверяет распределение классов"""
    logger.info("\n" + "=" * 80)
    logger.info("РАСПРЕДЕЛЕНИЕ КЛАССОВ")
    logger.info("=" * 80)
    
    train_dist = train_df[target_column].value_counts().sort_index()
    val_dist = val_df[target_column].value_counts().sort_index()
    test_dist = test_df[target_column].value_counts().sort_index()
    
    # Проценты
    train_pct = (train_dist / len(train_df) * 100).round(2)
    val_pct = (val_dist / len(val_df) * 100).round(2)
    test_pct = (test_dist / len(test_df) * 100).round(2)
    
    # Создаем таблицу сравнения
    comparison = pd.DataFrame({
        'Train_Count': train_dist,
        'Train_%': train_pct,
        'Val_Count': val_dist,
        'Val_%': val_pct,
        'Test_Count': test_dist,
        'Test_%': test_pct
    }).fillna(0)
    
    logger.info("\nСравнение распределения классов:")
    logger.info(comparison.to_string())
    
    # Проверка на значительные различия
    issues = []
    for class_idx in comparison.index:
        train_p = train_pct.get(class_idx, 0)
        val_p = val_pct.get(class_idx, 0)
        test_p = test_pct.get(class_idx, 0)
        
        # Проверяем разницу более 5%
        if abs(train_p - val_p) > 5:
            issues.append(f"⚠️  Класс {class_idx}: разница Train-Val = {abs(train_p - val_p):.2f}%")
        if abs(train_p - test_p) > 5:
            issues.append(f"⚠️  Класс {class_idx}: разница Train-Test = {abs(train_p - test_p):.2f}%")
        if abs(val_p - test_p) > 5:
            issues.append(f"⚠️  Класс {class_idx}: разница Val-Test = {abs(val_p - test_p):.2f}%")
    
    if issues:
        logger.warning("\nОбнаружены значительные различия в распределении:")
        for issue in issues[:10]:  # Показываем первые 10
            logger.warning(f"  {issue}")
        if len(issues) > 10:
            logger.warning(f"  ... и еще {len(issues) - 10} различий")
    else:
        logger.info("✓ Распределение классов схоже между выборками")
    
    # Статистический тест (Chi-square)
    if HAS_SCIPY:
        try:
            # Объединяем все классы
            all_classes = sorted(set(train_dist.index) | set(val_dist.index) | set(test_dist.index))
            observed = np.array([
                [train_dist.get(c, 0) for c in all_classes],
                [val_dist.get(c, 0) for c in all_classes],
                [test_dist.get(c, 0) for c in all_classes]
            ])
            
            chi2, p_value = stats.chi2_contingency(observed)[:2]
            logger.info(f"\nChi-square тест на независимость распределений:")
            logger.info(f"  Chi-square = {chi2:.4f}")
            logger.info(f"  p-value = {p_value:.6f}")
            
            if p_value < 0.05:
                logger.warning(f"  ⚠️  Распределения статистически различны (p < 0.05)")
            else:
                logger.info(f"  ✓ Распределения статистически схожи (p >= 0.05)")
        except Exception as e:
            logger.warning(f"  Не удалось выполнить Chi-square тест: {e}")
    else:
        logger.info("\n⚠️  Chi-square тест пропущен (scipy не установлен)")
    
    return {
        'train_dist': train_dist,
        'val_dist': val_dist,
        'test_dist': test_dist,
        'comparison': comparison,
        'has_issues': len(issues) > 0
    }

def check_feature_statistics(train_df, val_df, test_df, target_column='signal_class'):
    """Проверяет статистики фичей"""
    logger.info("\n" + "=" * 80)
    logger.info("СТАТИСТИКИ ФИЧЕЙ")
    logger.info("=" * 80)
    
    # Загружаем список исключенных фичей из файла
    excluded_features = []
    exclusions_file = Path('workspace/excluded_features.txt')
    if exclusions_file.exists():
        try:
            from utils.feature_exclusions import load_excluded_features
            excluded_features = load_excluded_features()
            if excluded_features:
                logger.info(f"Загружено {len(excluded_features)} фичей для исключения из excluded_features.txt")
        except Exception as e:
            logger.warning(f"Не удалось загрузить excluded_features.txt: {e}")
    
    # Получаем только фичи (исключаем target, служебные колонки и исключенные фичи)
    exclude_cols = [target_column, 'signal_class_name', 'max_future_return']
    exclude_cols.extend([col for col in train_df.columns if col.startswith('future_return')])
    exclude_cols.extend(excluded_features)  # Добавляем фичи из файла исключений
    
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    excluded_count = len(excluded_features)
    if excluded_count > 0:
        logger.info(f"Проверка {len(feature_cols)} фичей (исключено {excluded_count} из excluded_features.txt)...")
    else:
        logger.info(f"Проверка {len(feature_cols)} фичей...")
    
    # Вычисляем статистики
    train_stats = train_df[feature_cols].describe()
    val_stats = val_df[feature_cols].describe()
    test_stats = test_df[feature_cols].describe()
    
    # Проверяем различия в средних и стандартных отклонениях
    issues = []
    significant_diffs = []
    
    for col in feature_cols:
        train_mean = train_stats.loc['mean', col]
        val_mean = val_stats.loc['mean', col]
        test_mean = test_stats.loc['mean', col]
        
        train_std = train_stats.loc['std', col]
        val_std = val_stats.loc['std', col]
        test_std = test_stats.loc['std', col]
        
        # Проверка на значительные различия в средних (более 2 стандартных отклонений)
        if train_std > 0:
            train_val_diff = abs(train_mean - val_mean) / train_std
            train_test_diff = abs(train_mean - test_mean) / train_std
            
            if train_val_diff > 2:
                significant_diffs.append({
                    'feature': col,
                    'type': 'mean',
                    'split': 'train-val',
                    'diff_sigma': train_val_diff
                })
            if train_test_diff > 2:
                significant_diffs.append({
                    'feature': col,
                    'type': 'mean',
                    'split': 'train-test',
                    'diff_sigma': train_test_diff
                })
        
        # Проверка на значительные различия в std (более 50%)
        if train_std > 0:
            val_std_ratio = val_std / train_std
            test_std_ratio = test_std / train_std
            
            if val_std_ratio < 0.5 or val_std_ratio > 2.0:
                significant_diffs.append({
                    'feature': col,
                    'type': 'std',
                    'split': 'train-val',
                    'ratio': val_std_ratio
                })
            if test_std_ratio < 0.5 or test_std_ratio > 2.0:
                significant_diffs.append({
                    'feature': col,
                    'type': 'std',
                    'split': 'train-test',
                    'ratio': test_std_ratio
                })
    
    if significant_diffs:
        logger.warning(f"\nОбнаружено {len(significant_diffs)} фичей с значительными различиями:")
        for diff in significant_diffs[:20]:  # Показываем первые 20
            if diff['type'] == 'mean':
                logger.warning(f"  {diff['feature']}: {diff['split']} - разница = {diff['diff_sigma']:.2f}σ")
            else:
                logger.warning(f"  {diff['feature']}: {diff['split']} - std ratio = {diff['ratio']:.2f}")
        if len(significant_diffs) > 20:
            logger.warning(f"  ... и еще {len(significant_diffs) - 20} фичей")
    else:
        logger.info("✓ Статистики фичей схожи между выборками")
    
    # Сводная статистика
    logger.info(f"\nСводная статистика по всем фичам:")
    logger.info(f"  Train: mean_range = [{train_stats.loc['mean'].min():.4f}, {train_stats.loc['mean'].max():.4f}]")
    logger.info(f"  Val:   mean_range = [{val_stats.loc['mean'].min():.4f}, {val_stats.loc['mean'].max():.4f}]")
    logger.info(f"  Test:  mean_range = [{test_stats.loc['mean'].min():.4f}, {test_stats.loc['mean'].max():.4f}]")
    
    return {
        'train_stats': train_stats,
        'val_stats': val_stats,
        'test_stats': test_stats,
        'significant_diffs': significant_diffs,
        'has_issues': len(significant_diffs) > 0
    }

def create_visualizations(train_df, val_df, test_df, target_column='signal_class', save_dir='workspace/analysis-of-features'):
    """Создает визуализации для сравнения распределений"""
    if not HAS_PLOTTING:
        logger.warning("\n⚠️  matplotlib/seaborn не установлены. Визуализации пропущены.")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    logger.info("=" * 80)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Распределение классов
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    train_dist = train_df[target_column].value_counts().sort_index()
    val_dist = val_df[target_column].value_counts().sort_index()
    test_dist = test_df[target_column].value_counts().sort_index()
    
    axes[0].bar(train_dist.index, train_dist.values, alpha=0.7, label='Train')
    axes[0].set_title('Train: Распределение классов')
    axes[0].set_xlabel('Класс')
    axes[0].set_ylabel('Количество')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(val_dist.index, val_dist.values, alpha=0.7, color='orange', label='Val')
    axes[1].set_title('Val: Распределение классов')
    axes[1].set_xlabel('Класс')
    axes[1].set_ylabel('Количество')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].bar(test_dist.index, test_dist.values, alpha=0.7, color='green', label='Test')
    axes[2].set_title('Test: Распределение классов')
    axes[2].set_xlabel('Класс')
    axes[2].set_ylabel('Количество')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    class_dist_path = save_path / 'class_distribution_comparison.png'
    plt.savefig(class_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Сохранено: {class_dist_path}")
    
    # 2. Процентное распределение классов
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_classes = sorted(set(train_dist.index) | set(val_dist.index) | set(test_dist.index))
    train_pct = [(train_dist.get(c, 0) / len(train_df) * 100) for c in all_classes]
    val_pct = [(val_dist.get(c, 0) / len(val_df) * 100) for c in all_classes]
    test_pct = [(test_dist.get(c, 0) / len(test_df) * 100) for c in all_classes]
    
    x = np.arange(len(all_classes))
    width = 0.25
    
    ax.bar(x - width, train_pct, width, label='Train', alpha=0.7)
    ax.bar(x, val_pct, width, label='Val', alpha=0.7, color='orange')
    ax.bar(x + width, test_pct, width, label='Test', alpha=0.7, color='green')
    
    ax.set_xlabel('Класс')
    ax.set_ylabel('Процент (%)')
    ax.set_title('Процентное распределение классов по выборкам')
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    class_pct_path = save_path / 'class_distribution_percentage.png'
    plt.savefig(class_pct_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Сохранено: {class_pct_path}")
    
    # 3. Временное распределение
    fig, ax = plt.subplots(figsize=(14, 6))
    
    train_dates = train_df.index
    val_dates = val_df.index
    test_dates = test_df.index
    
    ax.scatter(train_dates, [1] * len(train_dates), alpha=0.3, s=1, label='Train', color='blue')
    ax.scatter(val_dates, [2] * len(val_dates), alpha=0.3, s=1, label='Val', color='orange')
    ax.scatter(test_dates, [3] * len(test_dates), alpha=0.3, s=1, label='Test', color='green')
    
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Train', 'Val', 'Test'])
    ax.set_xlabel('Дата')
    ax.set_title('Временное распределение данных')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    temporal_path = save_path / 'temporal_distribution.png'
    plt.savefig(temporal_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Сохранено: {temporal_path}")
    
    logger.info(f"\nВсе визуализации сохранены в: {save_path}")

def generate_report(train_df, val_df, test_df, temporal_info, class_info, feature_info, save_dir='workspace/analysis-of-features'):
    """Генерирует текстовый отчет"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    report_path = save_path / 'data_distribution_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЕТ О РАСПРЕДЕЛЕНИИ ДАННЫХ (TRAIN/VAL/TEST)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. ВРЕМЕННОЕ РАСПРЕДЕЛЕНИЕ\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train: {temporal_info['train_start']} - {temporal_info['train_end']} ({len(train_df)} образцов)\n")
        f.write(f"Val:   {temporal_info['val_start']} - {temporal_info['val_end']} ({len(val_df)} образцов)\n")
        f.write(f"Test:  {temporal_info['test_start']} - {temporal_info['test_end']} ({len(test_df)} образцов)\n")
        f.write(f"Перекрытия: {'Да' if temporal_info['has_overlap'] else 'Нет'}\n\n")
        
        f.write("2. РАСПРЕДЕЛЕНИЕ КЛАССОВ\n")
        f.write("-" * 80 + "\n")
        f.write(class_info['comparison'].to_string() + "\n\n")
        f.write(f"Проблемы: {'Да' if class_info['has_issues'] else 'Нет'}\n\n")
        
        f.write("3. СТАТИСТИКИ ФИЧЕЙ\n")
        f.write("-" * 80 + "\n")
        f.write(f"Проверено фичей: {len(feature_info['train_stats'].columns)}\n")
        f.write(f"Фичей с проблемами: {len(feature_info['significant_diffs'])}\n\n")
        
        if feature_info['significant_diffs']:
            f.write("Фичи с значительными различиями:\n")
            for diff in feature_info['significant_diffs'][:50]:
                if diff['type'] == 'mean':
                    f.write(f"  {diff['feature']}: {diff['split']} - разница = {diff['diff_sigma']:.2f}σ\n")
                else:
                    f.write(f"  {diff['feature']}: {diff['split']} - std ratio = {diff['ratio']:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("РЕКОМЕНДАЦИИ\n")
        f.write("=" * 80 + "\n")
        
        if temporal_info['has_overlap']:
            f.write("⚠️  Обнаружены временные перекрытия. Рекомендуется пересоздать разделение данных.\n")
        
        if class_info['has_issues']:
            f.write("⚠️  Обнаружены различия в распределении классов. Рассмотрите использование стратификации.\n")
        
        if feature_info['has_issues']:
            f.write("⚠️  Обнаружены различия в статистиках фичей. Это может указывать на:\n")
            f.write("   - Различные рыночные условия в разных периодах\n")
            f.write("   - Проблемы с нормализацией данных\n")
            f.write("   - Необходимость пересмотра разделения данных\n")
        
        if not temporal_info['has_overlap'] and not class_info['has_issues'] and not feature_info['has_issues']:
            f.write("✓ Распределение данных выглядит корректным.\n")
    
    logger.info(f"\nОтчет сохранен: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Проверка распределения данных в train/val/test выборках',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--save-dir',
        type=str,
        default='workspace/analysis-of-features',
        help='Директория для сохранения результатов (по умолчанию: workspace/analysis-of-features)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Не создавать графики'
    )
    
    args = parser.parse_args()
    
    try:
        # Загрузка данных
        train_df, val_df, test_df = load_datasets()
        
        # Проверки
        temporal_info = check_temporal_distribution(train_df, val_df, test_df)
        class_info = check_class_distribution(train_df, val_df, test_df)
        feature_info = check_feature_statistics(train_df, val_df, test_df)
        
        # Визуализации
        if not args.no_plots and HAS_PLOTTING:
            create_visualizations(train_df, val_df, test_df, save_dir=args.save_dir)
        elif not args.no_plots and not HAS_PLOTTING:
            logger.warning("\n⚠️  Визуализации пропущены (matplotlib/seaborn не установлены)")
        
        # Отчет
        generate_report(train_df, val_df, test_df, temporal_info, class_info, feature_info, save_dir=args.save_dir)
        
        # Итоговая сводка
        logger.info("\n" + "=" * 80)
        logger.info("ИТОГОВАЯ СВОДКА")
        logger.info("=" * 80)
        
        issues_count = sum([
            temporal_info['has_overlap'],
            class_info['has_issues'],
            feature_info['has_issues']
        ])
        
        if issues_count == 0:
            logger.info("✓ Все проверки пройдены успешно!")
            logger.info("  Распределение данных выглядит корректным.")
        else:
            logger.warning(f"⚠️  Обнаружено проблем: {issues_count}")
            if temporal_info['has_overlap']:
                logger.warning("  - Временные перекрытия")
            if class_info['has_issues']:
                logger.warning("  - Различия в распределении классов")
            if feature_info['has_issues']:
                logger.warning("  - Различия в статистиках фичей")
            logger.info("\n  Рекомендуется просмотреть детальный отчет и визуализации.")
        
        logger.info(f"\nРезультаты сохранены в: {args.save_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Ошибка при проверке распределения: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())

