"""
Утилита для валидации и проверки соответствия фичей между обучением и применением
"""
import pandas as pd
import pickle
from typing import Dict, List, Optional, Set
from pathlib import Path

def load_scaler_metadata(scaler_path: str) -> Dict:
    """
    Загружает метаданные из scaler файла
    
    Args:
        scaler_path: Путь к файлу scaler
    
    Returns:
        Словарь с метаданными
    """
    with open(scaler_path, 'rb') as f:
        data = pickle.load(f)
        return data.get('metadata', {})

def get_feature_list_from_scaler(scaler_path: str) -> List[str]:
    """
    Получает список фичей из scaler файла
    
    Args:
        scaler_path: Путь к файлу scaler
    
    Returns:
        Список названий фичей
    """
    with open(scaler_path, 'rb') as f:
        data = pickle.load(f)
        return data.get('feature_columns', [])

def validate_dataframe_features(df: pd.DataFrame, scaler_path: str, 
                               target_column: str = 'signal_class',
                               exclude_columns: Optional[List[str]] = None) -> Dict:
    """
    Валидирует соответствие фичей в DataFrame сохраненным фичам из scaler
    
    Args:
        df: DataFrame для проверки
        scaler_path: Путь к scaler файлу
        target_column: Название целевой переменной
        exclude_columns: Дополнительные колонки для исключения
    
    Returns:
        Словарь с результатами валидации:
        - 'is_valid': bool - все ли фичи присутствуют
        - 'missing_features': List[str] - отсутствующие фичи
        - 'extra_features': List[str] - лишние фичи
        - 'metadata': Dict - метаданные из scaler
    """
    # Загружаем сохраненные фичи
    saved_features = get_feature_list_from_scaler(scaler_path)
    metadata = load_scaler_metadata(scaler_path)
    
    # Получаем фичи из DataFrame
    exclude_patterns = [
        'target', 'label', 'direction', 'future_return', 
        'future_volatility', 'signal_class_name', 'max_future_return'
    ]
    if exclude_columns is None:
        exclude_columns = []
    
    df_features = [
        col for col in df.columns
        if col != target_column
        and not any(pattern in col.lower() for pattern in exclude_patterns)
        and col not in exclude_columns
    ]
    
    # Проверяем соответствие
    missing_features = sorted(list(set(saved_features) - set(df_features)))
    extra_features = sorted(list(set(df_features) - set(saved_features)))
    
    return {
        'is_valid': len(missing_features) == 0,
        'missing_features': missing_features,
        'extra_features': extra_features,
        'saved_features_count': len(saved_features),
        'df_features_count': len(df_features),
        'metadata': metadata
    }

def print_validation_report(validation_result: Dict):
    """
    Выводит отчет о валидации фичей
    
    Args:
        validation_result: Результат validate_dataframe_features
    """
    print("=" * 80)
    print("ОТЧЕТ О ВАЛИДАЦИИ ФИЧЕЙ")
    print("=" * 80)
    
    metadata = validation_result.get('metadata', {})
    if metadata:
        print(f"\nМетаданные из scaler:")
        if 'training_months' in metadata:
            print(f"  Месяцев данных при обучении: {metadata['training_months']}")
        if 'model_type' in metadata:
            print(f"  Тип модели: {metadata['model_type']}")
        if 'num_features' in metadata:
            print(f"  Количество фичей: {metadata['num_features']}")
        if 'preparation_config' in metadata:
            prep_config = metadata['preparation_config']
            if prep_config.get('remove_correlated_features'):
                print(f"  Удаление коррелированных фичей: Да")
                print(f"  Порог корреляции: {prep_config.get('correlation_threshold', 0.95)}")
            else:
                print(f"  Удаление коррелированных фичей: Нет")
        if 'saved_at' in metadata:
            print(f"  Дата сохранения: {metadata['saved_at']}")
    
    print(f"\nСтатистика фичей:")
    print(f"  Сохранено фичей: {validation_result['saved_features_count']}")
    print(f"  Фичей в DataFrame: {validation_result['df_features_count']}")
    
    if validation_result['is_valid']:
        print(f"\n✓ Валидация пройдена: все фичи присутствуют")
    else:
        print(f"\n❌ Валидация провалена: отсутствуют фичи")
        missing = validation_result['missing_features']
        print(f"  Отсутствует фичей: {len(missing)}")
        if len(missing) <= 20:
            for feat in missing:
                print(f"    - {feat}")
        else:
            for feat in missing[:20]:
                print(f"    - {feat}")
            print(f"    ... и еще {len(missing) - 20} фичей")
    
    extra = validation_result['extra_features']
    if extra:
        print(f"\n⚠️  Лишние фичи в DataFrame: {len(extra)}")
        if len(extra) <= 10:
            for feat in extra:
                print(f"    - {feat}")
        else:
            for feat in extra[:10]:
                print(f"    - {feat}")
            print(f"    ... и еще {len(extra) - 10} фичей")
        print(f"  Эти фичи будут проигнорированы при использовании модели.")
    
    print("=" * 80)

def compare_scalers(scaler_path1: str, scaler_path2: str):
    """
    Сравнивает два scaler файла
    
    Args:
        scaler_path1: Путь к первому scaler
        scaler_path2: Путь ко второму scaler
    """
    features1 = set(get_feature_list_from_scaler(scaler_path1))
    features2 = set(get_feature_list_from_scaler(scaler_path2))
    metadata1 = load_scaler_metadata(scaler_path1)
    metadata2 = load_scaler_metadata(scaler_path2)
    
    print("=" * 80)
    print("СРАВНЕНИЕ SCALER ФАЙЛОВ")
    print("=" * 80)
    
    print(f"\nScaler 1: {scaler_path1}")
    print(f"  Фичей: {len(features1)}")
    if metadata1:
        print(f"  Метаданные: {metadata1.get('model_type', 'N/A')}, {metadata1.get('training_months', 'N/A')} месяцев")
    
    print(f"\nScaler 2: {scaler_path2}")
    print(f"  Фичей: {len(features2)}")
    if metadata2:
        print(f"  Метаданные: {metadata2.get('model_type', 'N/A')}, {metadata2.get('training_months', 'N/A')} месяцев")
    
    common = features1 & features2
    only1 = features1 - features2
    only2 = features2 - features1
    
    print(f"\nСравнение:")
    print(f"  Общих фичей: {len(common)}")
    print(f"  Только в scaler 1: {len(only1)}")
    print(f"  Только в scaler 2: {len(only2)}")
    
    if only1:
        print(f"\n  Фичи только в scaler 1 (первые 10):")
        for feat in sorted(list(only1))[:10]:
            print(f"    - {feat}")
    
    if only2:
        print(f"\n  Фичи только в scaler 2 (первые 10):")
        for feat in sorted(list(only2))[:10]:
            print(f"    - {feat}")
    
    print("=" * 80)

