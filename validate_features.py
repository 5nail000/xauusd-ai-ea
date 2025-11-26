"""
Скрипт для валидации соответствия фичей между подготовленными данными и scaler
"""
import pandas as pd
import argparse
from utils.feature_validator import validate_dataframe_features, print_validation_report

def main():
    parser = argparse.ArgumentParser(
        description='Валидация соответствия фичей между данными и scaler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python validate_features.py                              # Валидация test данных
  python validate_features.py --data workspace/prepared/features/gold_train.csv  # Валидация train данных
  python validate_features.py --scaler workspace/prepared/scalers/feature_scaler_encoder.pkl
        """
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='workspace/prepared/features/gold_test.csv',
        help='Путь к CSV файлу с данными (по умолчанию: workspace/prepared/features/gold_test.csv)'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        default='workspace/prepared/scalers/feature_scaler_encoder.pkl',
        help='Путь к scaler файлу (по умолчанию: workspace/prepared/scalers/feature_scaler_encoder.pkl)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default='signal_class',
        help='Название целевой переменной (по умолчанию: signal_class)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ВАЛИДАЦИЯ СООТВЕТСТВИЯ ФИЧЕЙ")
    print("=" * 80)
    
    # Загрузка данных
    print(f"\n1. Загрузка данных из {args.data}...")
    try:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        print(f"   Загружено {len(df)} строк, {len(df.columns)} колонок")
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл {args.data} не найден")
        return
    
    # Валидация
    print(f"\n2. Валидация фичей с scaler {args.scaler}...")
    try:
        result = validate_dataframe_features(
            df=df,
            scaler_path=args.scaler,
            target_column=args.target
        )
        
        # Вывод отчета
        print_validation_report(result)
        
        # Возвращаем код выхода
        if result['is_valid']:
            print("\n✓ Валидация успешна!")
            return 0
        else:
            print("\n❌ Валидация провалена!")
            return 1
            
    except FileNotFoundError:
        print(f"❌ Ошибка: Файл scaler {args.scaler} не найден")
        print(f"   Убедитесь, что модель была обучена и scaler сохранен.")
        return 1
    except Exception as e:
        print(f"❌ Ошибка при валидации: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())

