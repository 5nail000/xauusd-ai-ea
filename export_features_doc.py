"""
Скрипт для экспорта документации по фичам
Можно запустить отдельно для существующей модели
"""
import argparse
import pickle
from pathlib import Path
from utils.feature_documentation import create_feature_documentation

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
  python export_features_doc.py --scaler models/feature_scaler_encoder.pkl
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
        scaler_path = f'models/feature_scaler_{args.model_type}.pkl'
    else:
        scaler_path = 'models/feature_scaler_encoder.pkl'
    
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
        
        print(f"✓ Найдено {len(feature_columns)} фичей")
    
    except Exception as e:
        print(f"❌ Ошибка при загрузке scaler: {e}")
        return 1
    
    # Определяем путь для сохранения
    if args.output:
        output_path = args.output
    elif args.model_type:
        output_path = f'models/checkpoints/{args.model_type}_model_features_documentation'
    else:
        output_path = 'models/checkpoints/features_documentation'
    
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

