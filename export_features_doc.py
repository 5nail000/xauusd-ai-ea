"""
Скрипт для экспорта документации по фичам
Можно запустить отдельно для существующей модели
Применяет список исключений из workspace/excluded_features.txt
"""
import argparse
import pickle
from pathlib import Path
from utils.feature_documentation import create_feature_documentation
from utils.feature_exclusions import load_excluded_features

def main():
    parser = argparse.ArgumentParser(
        description='Экспорт документации по фичам для модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Экспорт для encoder модели
  python export_features_doc.py --model-type encoder
  
  # Экспорт для timeseries модели
  python export_features_doc.py --model-type timeseries
  
  # С указанием пути к scaler
  python export_features_doc.py --scaler workspace/prepared/scalers/feature_scaler_encoder.pkl
        """
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['encoder', 'timeseries'],
        help='Тип модели'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        help='Путь к scaler файлу'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Путь для сохранения документации (без расширения)'
    )
    
    args = parser.parse_args()
    
    # Определяем scaler путь
    if args.scaler:
        scaler_path = args.scaler
    elif args.model_type:
        scaler_path = f'workspace/prepared/scalers/feature_scaler_{args.model_type}.pkl'
    else:
        scaler_path = 'workspace/prepared/scalers/feature_scaler_encoder.pkl'
    
    if not Path(scaler_path).exists():
        print(f"❌ Ошибка: Файл scaler не найден: {scaler_path}")
        return 1
    
    # Загружаем feature_columns из scaler
    print(f"Загрузка фичей из {scaler_path}...")
    try:
        with open(scaler_path, 'rb') as f:
            scaler_data = pickle.load(f)
            feature_columns = scaler_data.get('feature_columns', [])
        
        if not feature_columns:
            print("❌ Ошибка: В scaler не найдены feature_columns")
            return 1
        
        print(f"✓ Найдено {len(feature_columns)} фичей в scaler")
    
    except Exception as e:
        print(f"❌ Ошибка при загрузке scaler: {e}")
        return 1
    
    # Загружаем список исключенных фичей
    print(f"\nЗагрузка списка исключений...")
    excluded_features = load_excluded_features()
    
    if excluded_features:
        print(f"✓ Загружено {len(excluded_features)} фичей для исключения")
        # Фильтруем фичи, убирая исключенные
        original_count = len(feature_columns)
        feature_columns = [f for f in feature_columns if f not in excluded_features]
        excluded_count = original_count - len(feature_columns)
        
        if excluded_count > 0:
            print(f"✓ Исключено {excluded_count} фичей из списка")
            print(f"✓ Актуальных фичей: {len(feature_columns)}")
        else:
            print(f"✓ Все фичи из scaler актуальны (нет совпадений с исключениями)")
    else:
        print(f"✓ Список исключений не найден или пуст, используются все фичи из scaler")
    
    # Определяем путь для сохранения
    if args.output:
        output_path = args.output
    elif args.model_type:
        output_path = f'workspace/models/checkpoints/{args.model_type}_model_features_documentation'
    else:
        output_path = 'workspace/models/checkpoints/features_documentation'
    
    # Создаем документацию
    print(f"\nСоздание документации...")
    documentation = create_feature_documentation(
        feature_columns=feature_columns,
        scaler_path=scaler_path,
        output_path=output_path
    )
    
    print(f"\n✓ Документация создана:")
    print(f"  - {output_path}.json")
    print(f"  - {output_path}.md")
    
    return 0

if __name__ == '__main__':
    exit(main())

