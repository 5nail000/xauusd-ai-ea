"""
Единый скрипт для полного цикла: подготовка данных → обучение → Walk-Forward Validation → бэктестинг
"""
import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import timedelta

def calculate_walk_forward_params(data_days: int) -> dict:
    """
    Вычисляет оптимальные параметры Walk-Forward Validation на основе доступного периода данных
    
    Args:
        data_days: Общее количество дней данных
    
    Returns:
        Словарь с параметрами: train_days, val_days, test_days, step_days
    """
    # Минимальные требования для одного fold'а
    min_train = 30  # Минимум 30 дней для обучения
    min_val = 7      # Минимум 7 дней для валидации
    min_test = 7     # Минимум 7 дней для тестирования
    min_step = 5     # Минимум 5 дней шаг
    
    # Максимальные значения (чтобы не было слишком больших окон)
    max_train = 120  # Максимум 120 дней для обучения
    max_val = 30     # Максимум 30 дней для валидации
    max_test = 30    # Максимум 30 дней для тестирования
    max_step = 30    # Максимум 30 дней шаг
    
    # Целевое количество fold'ов (минимум 3, оптимально 5-10)
    target_folds = max(3, min(10, data_days // 30))
    
    # Вычисляем размер одного полного окна (train + val + test)
    window_size = data_days / target_folds
    
    # Train window: 50-60% от доступного периода, но в пределах min/max
    train_days = int(window_size * 0.55)
    train_days = max(min_train, min(train_days, max_train, data_days // 3))
    
    # Val window: 15-20% от доступного периода
    val_days = int(window_size * 0.17)
    val_days = max(min_val, min(val_days, max_val))
    
    # Test window: аналогично val
    test_days = int(window_size * 0.17)
    test_days = max(min_test, min(test_days, max_test))
    
    # Step size: 10-15% от доступного периода
    step_days = int(window_size * 0.12)
    step_days = max(min_step, min(step_days, max_step))
    
    # Проверяем, что получается достаточно fold'ов
    total_window = train_days + val_days + test_days
    available_for_folds = data_days - total_window
    estimated_folds = max(1, available_for_folds // step_days + 1)
    
    # Если fold'ов слишком мало, уменьшаем окна
    if estimated_folds < 3:
        # Уменьшаем пропорционально
        scale = 0.8
        train_days = max(min_train, int(train_days * scale))
        val_days = max(min_val, int(val_days * scale))
        test_days = max(min_test, int(test_days * scale))
        step_days = max(min_step, int(step_days * scale))
    
    # Округляем до целых значений
    train_days = int(train_days)
    val_days = int(val_days)
    test_days = int(test_days)
    step_days = int(step_days)
    
    return {
        'train_days': train_days,
        'val_days': val_days,
        'test_days': test_days,
        'step_days': step_days
    }

def run_command(command, description):
    """
    Запускает команду и обрабатывает ошибки
    
    Args:
        command: Список аргументов для subprocess.run
        description: Описание команды для вывода
    
    Returns:
        True если успешно, False если ошибка
    """
    print("\n" + "=" * 80)
    print(f"{description}")
    print("=" * 80)
    print(f"Команда: {' '.join(command)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"\n✓ {description} завершено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при выполнении: {description}")
        print(f"   Код возврата: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Прервано пользователем: {description}")
        return False
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        return False

def check_data_files():
    """Проверяет наличие подготовленных данных"""
    required_files = [
        'workspace/prepared/features/gold_train.csv',
        'workspace/prepared/features/gold_val.csv',
        'workspace/prepared/features/gold_test.csv'
    ]
    
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        print("✓ Подготовленные данные найдены")
        return True
    else:
        missing = [f for f in required_files if not os.path.exists(f)]
        print(f"⚠️  Отсутствуют файлы: {', '.join(missing)}")
        return False

def check_model_files(model_type='encoder'):
    """Проверяет наличие обученных моделей"""
    required_files = [
        f'workspace/models/checkpoints/{model_type}_model.pth',
        f'workspace/prepared/scalers/feature_scaler_{model_type}.pkl'
    ]
    
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        print(f"✓ Модель {model_type} найдена")
        return True
    else:
        missing = [f for f in required_files if not os.path.exists(f)]
        print(f"⚠️  Отсутствуют файлы модели {model_type}: {', '.join(missing)}")
        return False

def main():
    """Главная функция для запуска полного цикла"""
    parser = argparse.ArgumentParser(
        description='Полный цикл: подготовка данных -> обучение -> бэктестинг',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Полный цикл с параметрами по умолчанию
  python full_pipeline.py
  
  # 12 месяцев данных, только encoder модель
  python full_pipeline.py --months 12 --encoder-only
  
  # 30 дней данных
  python full_pipeline.py --days 30
  
  # 1 месяц данных
  python full_pipeline.py --months 1
  
  # Пропустить подготовку данных (если уже есть)
  python full_pipeline.py --skip-prepare
  
  # Пропустить обучение (если модель уже обучена)
  python full_pipeline.py --skip-train
  
  # Только бэктестинг
  python full_pipeline.py --skip-prepare --skip-train
  
  # С настройками обучения
  python full_pipeline.py --months 12 --epochs 50 --batch-size 16
  
  # С оптимизацией фичей (объединенный анализ)
  python full_pipeline.py --months 12 --remove-correlated
  
  # С оптимизацией фичей и кастомным порогом корреляции
  python full_pipeline.py --months 12 --remove-correlated --correlation-threshold 0.90
  
  # Режим offline (без подключения к MT5, только кэшированные данные)
  python full_pipeline.py --offline --days 30
        """
    )
    
    # Параметры подготовки данных
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=None,
        help='Количество месяцев данных (по умолчанию: 12, если --days не указан)'
    )
    
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=None,
        help='Количество дней данных (приоритет над --months)'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default='XAUUSD',
        help='Торговый символ (по умолчанию: XAUUSD)'
    )
    
    parser.add_argument(
        '--no-ticks',
        action='store_true',
        help='Не загружать тиковые данные'
    )
    
    parser.add_argument(
        '--no-higher-tf',
        action='store_true',
        help='Не загружать старшие таймфреймы'
    )
    
    parser.add_argument(
        '--force-prepare',
        action='store_true',
        help='Принудительно регенерировать данные'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Не использовать кэш при подготовке данных'
    )
    
    parser.add_argument(
        '--offline',
        action='store_true',
        help='Режим offline - работа только с кэшированными данными без подключения к MT5'
    )
    
    parser.add_argument(
        '--remove-correlated',
        action='store_true',
        help='Выполнить объединенный анализ фичей и создать список исключений (использует analyze_and_exclude_features.py)'
    )
    
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.95,
        help='Порог корреляции для исключения фичей (по умолчанию: 0.95)'
    )
    
    parser.add_argument(
        '--analyze-features',
        action='store_true',
        help='Выполнить объединенный анализ фичей (аналогично --remove-correlated, включает все типы анализа)'
    )
    
    parser.add_argument(
        '--generate-feature-plots',
        action='store_true',
        help='[УСТАРЕЛО] Параметр больше не используется (графики не генерируются в объединенном скрипте)'
    )
    
    parser.add_argument(
        '--save-detailed-analyze',
        action='store_true',
        help='Сохранить детальные результаты анализа фичей в workspace/analysis-of-features/ (CSV файлы и HTML отчет)'
    )
    
    # Параметры обучения
    parser.add_argument(
        '--encoder-only',
        action='store_true',
        help='Обучить только encoder модель'
    )
    
    parser.add_argument(
        '--timeseries-only',
        action='store_true',
        help='Обучить только timeseries модель'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Размер батча (по умолчанию: 32)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Количество эпох (по умолчанию: 100)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Терпение для early stopping (по умолчанию: 10)'
    )
    
    parser.add_argument(
        '--no-class-weights',
        action='store_true',
        help='НЕ использовать веса классов (по умолчанию веса включены)'
    )
    
    parser.add_argument(
        '--class-weight-method',
        type=str,
        default='balanced',
        choices=['balanced', 'inverse', 'sqrt'],
        help='Метод вычисления весов классов (по умолчанию: balanced)'
    )
    
    # Параметры бэктестинга
    parser.add_argument(
        '--model-type',
        type=str,
        default='encoder',
        choices=['encoder', 'timeseries'],
        help='Тип модели для бэктестинга (по умолчанию: encoder)'
    )
    
    # Пропуск этапов
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Пропустить подготовку данных'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Пропустить обучение моделей'
    )
    
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Пропустить бэктестинг'
    )
    
    parser.add_argument(
        '--use-walk-forward',
        action='store_true',
        help='Использовать Walk-Forward Validation вместо обычного обучения (альтернативный метод)'
    )
    parser.add_argument(
        '--walk-forward-train-days',
        type=int,
        default=None,
        help='Размер обучающего окна для Walk-Forward Validation (в днях, по умолчанию вычисляется автоматически)'
    )
    parser.add_argument(
        '--walk-forward-val-days',
        type=int,
        default=None,
        help='Размер валидационного окна для Walk-Forward Validation (в днях, по умолчанию вычисляется автоматически)'
    )
    parser.add_argument(
        '--walk-forward-test-days',
        type=int,
        default=None,
        help='Размер тестового окна для Walk-Forward Validation (в днях, по умолчанию вычисляется автоматически)'
    )
    parser.add_argument(
        '--walk-forward-step-days',
        type=int,
        default=None,
        help='Шаг сдвига окна для Walk-Forward Validation (в днях, по умолчанию вычисляется автоматически)'
    )
    
    args = parser.parse_args()
    
    # Определяем период для отображения
    if args.days is not None:
        period_str = f"{args.days} дней"
        period_label = f"{args.days}d"
    elif args.months is not None:
        period_str = f"{args.months} месяцев"
        period_label = f"{args.months}m"
    else:
        period_str = "12 месяцев (по умолчанию)"
        period_label = "12m"
        args.months = 12
    
    # Вывод параметров
    print("\n" + "=" * 80)
    print("ПОЛНЫЙ ЦИКЛ: Подготовка -> Обучение -> Бэктестинг")
    print("=" * 80)
    print(f"\nПараметры:")
    print(f"  Период данных: {period_str}")
    print(f"  Символ: {args.symbol}")
    print(f"  Тики: {'Нет' if args.no_ticks else 'Да'}")
    print(f"  Старшие таймфреймы: {'Нет' if args.no_higher_tf else 'Да'}")
    print(f"  Оптимизация фичей: {'Да' if (args.remove_correlated or args.analyze_features) else 'Нет'}")
    if args.remove_correlated or args.analyze_features:
        print(f"  Порог корреляции: {args.correlation_threshold}")
        print(f"  Используется: analyze_and_exclude_features.py")
        print(f"  Детальные результаты: {'Да' if args.save_detailed_analyze else 'Нет'}")
    print(f"  Размер батча: {args.batch_size}")
    print(f"  Эпох: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    use_class_weights = not args.no_class_weights
    print(f"  Веса классов: {'Да' if use_class_weights else 'Нет'}")
    if use_class_weights:
        print(f"  Метод весов: {args.class_weight_method}")
    print(f"  Метод обучения: {'Walk-Forward Validation' if args.use_walk_forward else 'Обычное обучение'}")
    if args.use_walk_forward:
        # Параметры будут вычислены автоматически, если не указаны
        if args.walk_forward_train_days is None:
            print(f"    Параметры: будут вычислены автоматически на основе доступных данных")
        else:
            print(f"    Окна: train={args.walk_forward_train_days}d, val={args.walk_forward_val_days or 'auto'}d, test={args.walk_forward_test_days or 'auto'}d")
            print(f"    Шаг: {args.walk_forward_step_days or 'auto'}d")
    print(f"  Модель для бэктестинга: {args.model_type}")
    print(f"\nЭтапы:")
    print(f"  Подготовка данных: {'Пропущено' if args.skip_prepare else 'Выполнится'}")
    print(f"  Обучение: {'Пропущено' if args.skip_train else 'Выполнится'}")
    print(f"  Бэктестинг: {'Пропущено' if args.skip_backtest else 'Выполнится'}")
    print("=" * 80)
    
    # Проверка наличия необходимых скриптов
    required_scripts = [
        'prepare_gold_data.py',
        'train_all_models.py',
        'backtest_strategy.py'
    ]
    
    missing_scripts = [s for s in required_scripts if not os.path.exists(s)]
    if missing_scripts:
        print(f"\n❌ Ошибка: Отсутствуют необходимые скрипты: {', '.join(missing_scripts)}")
        return 1
    
    # Этап 1: Подготовка данных
    if not args.skip_prepare:
        # Проверяем, есть ли уже данные
        if check_data_files() and not args.force_prepare:
            print("\n⚠️  Подготовленные данные уже существуют.")
            response = input("   Пропустить подготовку данных? (y/n): ").strip().lower()
            if response == 'y':
                print("   Пропускаем подготовку данных...")
                args.skip_prepare = True
            else:
                print("   Продолжаем подготовку данных...")
        
        if not args.skip_prepare:
            prepare_cmd = [
                sys.executable,
                'prepare_gold_data.py',
                '--symbol', args.symbol
            ]
            
            # Добавляем параметр периода (приоритет у days)
            if args.days is not None:
                prepare_cmd.extend(['--days', str(args.days)])
            elif args.months is not None:
                prepare_cmd.extend(['--months', str(args.months)])
            
            if args.no_ticks:
                prepare_cmd.append('--no-ticks')
            if args.no_higher_tf:
                prepare_cmd.append('--no-higher-tf')
            if args.force_prepare:
                prepare_cmd.append('--force')
            if args.no_cache:
                prepare_cmd.append('--no-cache')
            if args.offline:
                prepare_cmd.append('--offline')
            prepare_cmd.append('--no-ask')  # Не спрашивать при наличии данных
            
            if not run_command(prepare_cmd, "ЭТАП 1: Подготовка данных"):
                print("\n❌ Ошибка на этапе подготовки данных. Остановка.")
                return 1
            
            # Оптимизация фичей: объединенный анализ и исключение
            if args.remove_correlated or args.analyze_features:
                print("\n" + "=" * 80)
                print("ОПТИМИЗАЦИЯ ФИЧЕЙ: Объединенный анализ и исключение")
                print("=" * 80)
                print("Используется: analyze_and_exclude_features.py")
                print("Результат: workspace/excluded_features.txt")
                print("Фичи исключаются автоматически при создании DataLoader'ов")
                
                train_path = 'workspace/prepared/features/gold_train.csv'
                val_path = 'workspace/prepared/features/gold_val.csv'
                test_path = 'workspace/prepared/features/gold_test.csv'
                
                # Проверяем наличие всех файлов
                missing_files = []
                for name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
                    if not os.path.exists(path):
                        missing_files.append((name, path))
                
                if missing_files:
                    print(f"\n⚠️  Не найдены файлы:")
                    for name, path in missing_files:
                        print(f"   - {name}: {path}")
                    print("   Пропускаем оптимизацию фичей...")
                else:
                    analyze_cmd = [
                        sys.executable,
                        'analyze_and_exclude_features.py',
                        '--train', train_path,
                        '--val', val_path,
                        '--test', test_path
                    ]
                    
                    # Добавляем параметры, если указаны
                    if args.remove_correlated:
                        analyze_cmd.extend(['--correlation-threshold', str(args.correlation_threshold)])
                    
                    # Если указан только --analyze-features без --remove-correlated,
                    # выполняем полный анализ (включая корреляцию)
                    if args.analyze_features and not args.remove_correlated:
                        # По умолчанию выполняем полный анализ
                        pass
                    
                    # Добавляем параметр сохранения детальных результатов
                    if args.save_detailed_analyze:
                        analyze_cmd.append('--save-details')
                    
                    if not run_command(analyze_cmd, "Объединенный анализ и исключение фичей"):
                        print("\n⚠️  Анализ фичей завершился с предупреждениями, но продолжаем...")
                    else:
                        # Проверяем, был ли создан файл исключений
                        excluded_features_file = Path('workspace/excluded_features.txt')
                        if excluded_features_file.exists():
                            print(f"\n✓ Список фичей для исключения создан: {excluded_features_file}")
                            print("  Фичи будут автоматически исключены при создании DataLoader'ов")
                        else:
                            print(f"\n⚠️  Файл исключений не создан: {excluded_features_file}")
                        
                        # Проверяем детальные результаты
                        if args.save_detailed_analyze:
                            analysis_dir = Path('workspace/analysis-of-features')
                            if analysis_dir.exists():
                                print(f"\n✓ Детальные результаты сохранены в: {analysis_dir}")
                                print("  Доступны CSV файлы и HTML отчет")
    else:
        print("\n⏭️  Пропуск этапа подготовки данных")
        if not check_data_files():
            print("❌ Ошибка: Подготовленные данные не найдены!")
            print("   Запустите без --skip-prepare или подготовьте данные вручную.")
            return 1
    
    # Этап 2: Обучение моделей (обычное или Walk-Forward Validation)
    if args.use_walk_forward:
        # Используем Walk-Forward Validation вместо обычного обучения
        print("\n" + "=" * 80)
        print("ЭТАП 2: ОБУЧЕНИЕ МОДЕЛЕЙ (WALK-FORWARD VALIDATION)")
        print("=" * 80)
        
        try:
            import pandas as pd
            import torch
            from validation.walk_forward import WalkForwardValidator
            from models.model_factory import create_model, get_model_config
            from models.data_loader import create_dataloaders, compute_class_weights
            from models.trainer import ModelTrainer
            from models.evaluator import ModelEvaluator
            
            # Загружаем все данные для walk-forward validation
            print("\nЗагрузка данных для Walk-Forward Validation...")
            train_df = pd.read_csv('workspace/prepared/features/gold_train.csv', 
                                  index_col=0, parse_dates=True)
            val_df = pd.read_csv('workspace/prepared/features/gold_val.csv', 
                                index_col=0, parse_dates=True)
            test_df = pd.read_csv('workspace/prepared/features/gold_test.csv', 
                                 index_col=0, parse_dates=True)
            
            # Объединяем все данные для walk-forward validation
            all_data = pd.concat([train_df, val_df, test_df]).sort_index()
            print(f"  Загружено {len(all_data)} образцов")
            print(f"  Период: {all_data.index[0]} - {all_data.index[-1]}")
            
            # Вычисляем период данных в днях
            if isinstance(all_data.index, pd.DatetimeIndex):
                data_period_days = (all_data.index[-1] - all_data.index[0]).days
            else:
                # Если нет DatetimeIndex, используем количество образцов как приближение
                data_period_days = len(all_data) // 1440
            
            # Вычисляем параметры Walk-Forward Validation, если не указаны
            if args.walk_forward_train_days is None:
                print(f"\n  Вычисление оптимальных параметров Walk-Forward Validation...")
                print(f"  Доступный период данных: {data_period_days} дней")
                
                wf_params = calculate_walk_forward_params(data_period_days)
                train_days = wf_params['train_days']
                val_days = wf_params['val_days']
                test_days = wf_params['test_days']
                step_days = wf_params['step_days']
                
                # Оцениваем количество fold'ов
                total_window = train_days + val_days + test_days
                available_for_folds = data_period_days - total_window
                estimated_folds = max(1, available_for_folds // step_days + 1)
                
                print(f"  Вычисленные параметры:")
                print(f"    Train window: {train_days} дней")
                print(f"    Val window: {val_days} дней")
                print(f"    Test window: {test_days} дней")
                print(f"    Step size: {step_days} дней")
                print(f"    Ожидаемое количество fold'ов: ~{estimated_folds}")
            else:
                # Используем указанные пользователем параметры
                train_days = args.walk_forward_train_days
                val_days = args.walk_forward_val_days or 15
                test_days = args.walk_forward_test_days or 15
                step_days = args.walk_forward_step_days or 15
                print(f"\n  Используются указанные параметры:")
                print(f"    Train window: {train_days} дней")
                print(f"    Val window: {val_days} дней")
                print(f"    Test window: {test_days} дней")
                print(f"    Step size: {step_days} дней")
            
            # Функция обучения для одного окна
            def train_model_for_fold(train_df_fold, val_df_fold, **kwargs):
                model_type = kwargs.get('model_type', args.model_type)
                batch_size = kwargs.get('batch_size', args.batch_size)
                num_epochs = kwargs.get('num_epochs', args.epochs)
                use_class_weights_fold = kwargs.get('use_class_weights', use_class_weights)
                class_weight_method = kwargs.get('class_weight_method', args.class_weight_method)
                
                # Создаем DataLoader'ы
                # exclude_columns будет загружен автоматически из excluded_features.txt внутри create_dataloaders
                train_loader, val_loader, _, seq_gen = create_dataloaders(
                    train_df=train_df_fold,
                    val_df=val_df_fold,
                    test_df=val_df_fold,
                    sequence_length=60,
                    batch_size=batch_size,
                    target_column='signal_class'
                )
                
                num_features = train_loader.dataset.sequences.shape[2]
                
                # Вычисляем веса классов
                class_weights = None
                if use_class_weights_fold:
                    class_weights = compute_class_weights(
                        train_df_fold,
                        target_column='signal_class',
                        method=class_weight_method
                    )
                
                # Создаем модель
                config = get_model_config(
                    model_type=model_type,
                    num_features=num_features,
                    num_classes=5,
                    sequence_length=60,
                    d_model=256,
                    n_layers=4 if model_type == 'encoder' else 6,
                    n_heads=8,
                    dropout=0.1
                )
                model = create_model(config)
                
                # Создаем trainer
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trainer = ModelTrainer(
                    model=model,
                    device=device,
                    learning_rate=1e-4,
                    weight_decay=1e-5,
                    scheduler_type='cosine',
                    model_config=config,
                    model_type=model_type,
                    use_class_weights=use_class_weights_fold,
                    class_weights=class_weights
                )
                
                # Обучаем модель
                checkpoint_path = f'workspace/models/checkpoints/walkforward_{model_type}_fold_temp.pth'
                trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs,
                    early_stopping_patience=args.patience,
                    checkpoint_path=checkpoint_path,
                    save_history=False  # Не сохраняем историю для каждого fold
                )
                
                return model
            
            # Функция оценки для одного окна
            def evaluate_model_for_fold(model, test_df_fold, **kwargs):
                model_type = kwargs.get('model_type', args.model_type)
                batch_size = kwargs.get('batch_size', args.batch_size)
                
                # Создаем DataLoader для теста
                # exclude_columns будет загружен автоматически из excluded_features.txt внутри create_dataloaders
                _, _, test_loader, _ = create_dataloaders(
                    train_df=test_df_fold.iloc[:100],  # Заглушка
                    val_df=test_df_fold.iloc[:100],    # Заглушка
                    test_df=test_df_fold,
                    sequence_length=60,
                    batch_size=batch_size,
                    target_column='signal_class'
                )
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                evaluator = ModelEvaluator(model, device)
                metrics = evaluator.evaluate(test_loader)
                
                return metrics
            
            # Создаем validator
            validator = WalkForwardValidator(
                train_size=train_days,
                val_size=val_days,
                test_size=test_days,
                step_size=step_days,
                use_days=True  # Используем дни, не образцы
            )
            
            # Выполняем walk-forward validation
            print("\nВыполнение Walk-Forward Validation...")
            print("   Это может занять некоторое время...\n")
            
            results = validator.validate(
                df=all_data,
                train_function=train_model_for_fold,
                evaluate_function=evaluate_model_for_fold,
                model_type=args.model_type,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                use_class_weights=use_class_weights,
                class_weight_method=args.class_weight_method
            )
            
            # Выводим результаты
            print("\n" + "=" * 80)
            print("РЕЗУЛЬТАТЫ WALK-FORWARD VALIDATION")
            print("=" * 80)
            print(results['summary'])
            
            # Сохраняем детальные результаты
            results_dir = Path('workspace/results/walk_forward')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_df = pd.DataFrame(results['folds'])
            results_path = results_dir / 'walk_forward_results.csv'
            results_df.to_csv(results_path, index=False)
            print(f"\n✓ Детальные результаты сохранены: {results_path}")
            
            # Сохраняем последнюю модель для бэктестинга
            # (в реальности можно выбрать лучшую модель по метрикам)
            last_model_path = f'workspace/models/checkpoints/walkforward_{args.model_type}_fold_temp.pth'
            final_model_path = f'workspace/models/checkpoints/{args.model_type}_model.pth'
            
            if os.path.exists(last_model_path):
                shutil.copy2(last_model_path, final_model_path)
                print(f"\n✓ Последняя модель из Walk-Forward Validation сохранена: {final_model_path}")
                print(f"   Для бэктестинга будет использована эта модель.")
                print(f"   Рекомендуется выбрать лучшую модель на основе метрик из walk_forward_results.csv")
            else:
                print(f"\n⚠️  Модель не найдена: {last_model_path}")
                print(f"   Бэктестинг может быть невозможен.")
            
        except Exception as e:
            print(f"\n❌ Ошибка при выполнении Walk-Forward Validation: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    elif not args.skip_train:
        # Обычное обучение
        # Определяем, какие модели обучать
        if args.encoder_only:
            models_to_train = ['encoder']
        elif args.timeseries_only:
            models_to_train = ['timeseries']
        else:
            models_to_train = ['encoder', 'timeseries']
        
        # Проверяем, есть ли уже обученные модели
        existing_models = []
        for model_type in models_to_train:
            if check_model_files(model_type):
                existing_models.append(model_type)
        
        if existing_models and not args.force_prepare:
            print(f"\n⚠️  Обученные модели уже существуют: {', '.join(existing_models)}")
            response = input("   Переобучить модели? (y/n): ").strip().lower()
            if response != 'y':
                print("   Пропускаем обучение моделей...")
                args.skip_train = True
        
        if not args.skip_train:
            train_cmd = [
                sys.executable,
                'train_all_models.py',
                '--batch-size', str(args.batch_size),
                '--epochs', str(args.epochs),
                '--patience', str(args.patience)
            ]
            
            # Добавляем параметр периода (train_all_models.py использует только --months для метаданных)
            # Если указаны дни, не передаем --months (данные уже подготовлены с нужным периодом)
            # Если указаны месяцы, передаем для метаданных
            if args.months is not None:
                train_cmd.extend(['--months', str(args.months)])
            # Если указаны только дни, не передаем --months (будет использоваться значение по умолчанию 12)
            
            if args.encoder_only:
                train_cmd.append('--encoder-only')
            elif args.timeseries_only:
                train_cmd.append('--timeseries-only')
            
            if not args.no_class_weights:
                # По умолчанию веса включены, передаем только если нужно отключить
                pass  # Веса включены по умолчанию
            else:
                train_cmd.append('--no-class-weights')
            train_cmd.extend(['--class-weight-method', args.class_weight_method])
            
            if not run_command(train_cmd, "ЭТАП 2: Обучение моделей"):
                print("\n❌ Ошибка на этапе обучения. Продолжаем с бэктестингом...")
                # Проверяем, есть ли хотя бы одна модель для бэктестинга
                if not check_model_files(args.model_type):
                    print(f"❌ Модель {args.model_type} не найдена. Бэктестинг невозможен.")
                    return 1
    else:
        print("\n⏭️  Пропуск этапа обучения")
        if not check_model_files(args.model_type):
            print(f"❌ Ошибка: Модель {args.model_type} не найдена!")
            print("   Запустите без --skip-train или обучите модель вручную.")
            return 1
    
    # Этап 3: Бэктестинг
    if not args.skip_backtest:
        # Проверяем наличие модели для бэктестинга
        if not check_model_files(args.model_type):
            print(f"\n❌ Ошибка: Модель {args.model_type} не найдена для бэктестинга!")
            print("   Обучите модель или выберите другую с помощью --model-type")
            return 1
        
        # Бэктестинг запускается через Python API, а не через subprocess
        # (так как нужно передать параметры модели)
        print("\n" + "=" * 80)
        print("ЭТАП 4: БЭКТЕСТИНГ")
        print("=" * 80)
        
        try:
            import pandas as pd
            from trading.backtester import Backtester
            from config.trading_config import TradingConfig
            
            # Конфигурация торговли
            trading_config = TradingConfig(
                base_lot_size=0.1,
                take_profit_min=50.0,
                take_profit_max=130.0,
                stop_loss=100.0,
                use_trailing_stop=True,
                trailing_start=30.0,
                trailing_step=20.0,
                use_partial_close=True,
                partial_close_at=60.0,
                partial_close_ratio=0.5,
                use_signal_confidence=True,
                confidence_threshold=0.8,
                confidence_multiplier=1.5,
                spread_pips=2.0
            )
            
            # Параметры модели
            model_path = f'workspace/models/checkpoints/{args.model_type}_model.pth'
            scaler_path = f'workspace/prepared/scalers/feature_scaler_{args.model_type}.pkl'
            
            # Проверяем наличие scaler
            if not os.path.exists(scaler_path):
                scaler_path = 'workspace/prepared/scalers/feature_scaler.pkl'
                if not os.path.exists(scaler_path):
                    print(f"❌ Ошибка: Scaler не найден!")
                    return 1
            
            # Загрузка тестовых данных
            print("\n1. Загрузка данных...")
            test_df = pd.read_csv('workspace/prepared/features/gold_test.csv', index_col=0, parse_dates=True)
            print(f"   Загружено {len(test_df)} свечей")
            
            # Создание бэктестера
            print("\n2. Инициализация бэктестера...")
            backtester = Backtester(
                model_path=model_path,
                scaler_path=scaler_path,
                model_type=args.model_type,
                trading_config=trading_config
            )
            
            # Запуск бэктестинга
            print("\n3. Запуск бэктестинга...")
            results = backtester.backtest(test_df, start_idx=60)
            
            # Сохранение результатов
            print("\n4. Сохранение результатов...")
            os.makedirs('trading', exist_ok=True)
            
            results_df = pd.DataFrame([results])
            results_df.to_csv('trading/backtest_results.csv', index=False)
            
            if 'equity_history' in results:
                results['equity_history'].to_csv('trading/equity_history.csv', index=False)
                print("   История equity сохранена: trading/equity_history.csv")
            
            if backtester.position_manager.closed_positions:
                positions_df = pd.DataFrame(backtester.position_manager.closed_positions)
                positions_df.to_csv('trading/closed_positions.csv', index=False)
                print("   Закрытые позиции сохранены: trading/closed_positions.csv")
            
            print("\n✓ Бэктестинг завершен успешно")
            
        except Exception as e:
            print(f"\n❌ Ошибка при бэктестинге: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Пропуск этапа бэктестинга")
    
    # Итоговая сводка
    print("\n\n" + "=" * 80)
    print("ПОЛНЫЙ ЦИКЛ ЗАВЕРШЕН!")
    print("=" * 80)
    print("\nСозданные файлы:")
    print("\nДанные:")
    print("  - workspace/prepared/features/gold_train.csv")
    print("  - workspace/prepared/features/gold_val.csv")
    print("  - workspace/prepared/features/gold_test.csv")
    
    models_created = []
    if args.encoder_only or (not args.timeseries_only and not args.skip_train):
        models_created.append('encoder')
    if args.timeseries_only or (not args.encoder_only and not args.skip_train):
        models_created.append('timeseries')
    
    for model_type in models_created:
        print(f"\n{model_type.upper()} модель:")
        print(f"  - workspace/models/checkpoints/{model_type}_model.pth")
        print(f"  - workspace/prepared/scalers/feature_scaler_{model_type}.pkl")
        print(f"  - workspace/models/metrics/{model_type}_model_history.csv")
        print(f"  - workspace/models/metrics/{model_type}_model_training_curves.png")
        print(f"  - workspace/models/metrics/confusion_matrix_{model_type}.png")
    
    if not args.skip_backtest:
        print("\nБэктестинг:")
        print("  - trading/backtest_results.csv")
        print("  - trading/equity_history.csv")
        print("  - trading/closed_positions.csv")
    
    print("\n" + "=" * 80)
    return 0

if __name__ == '__main__':
    exit(main())

