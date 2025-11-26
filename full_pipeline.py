"""
Единый скрипт для полного цикла: подготовка данных → обучение → бэктестинг
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

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
  
  # Пропустить подготовку данных (если уже есть)
  python full_pipeline.py --skip-prepare
  
  # Пропустить обучение (если модель уже обучена)
  python full_pipeline.py --skip-train
  
  # Только бэктестинг
  python full_pipeline.py --skip-prepare --skip-train
  
  # С настройками обучения
  python full_pipeline.py --months 12 --epochs 50 --batch-size 16
        """
    )
    
    # Параметры подготовки данных
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=12,
        help='Количество месяцев данных (по умолчанию: 12)'
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
    
    args = parser.parse_args()
    
    # Вывод параметров
    print("\n" + "=" * 80)
    print("ПОЛНЫЙ ЦИКЛ: Подготовка -> Обучение -> Бэктестинг")
    print("=" * 80)
    print(f"\nПараметры:")
    print(f"  Месяцев данных: {args.months}")
    print(f"  Символ: {args.symbol}")
    print(f"  Тики: {'Нет' if args.no_ticks else 'Да'}")
    print(f"  Старшие таймфреймы: {'Нет' if args.no_higher_tf else 'Да'}")
    print(f"  Размер батча: {args.batch_size}")
    print(f"  Эпох: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
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
                '--months', str(args.months),
                '--symbol', args.symbol
            ]
            
            if args.no_ticks:
                prepare_cmd.append('--no-ticks')
            if args.no_higher_tf:
                prepare_cmd.append('--no-higher-tf')
            if args.force_prepare:
                prepare_cmd.append('--force')
            if args.no_cache:
                prepare_cmd.append('--no-cache')
            prepare_cmd.append('--no-ask')  # Не спрашивать при наличии данных
            
            if not run_command(prepare_cmd, "ЭТАП 1: Подготовка данных"):
                print("\n❌ Ошибка на этапе подготовки данных. Остановка.")
                return 1
    else:
        print("\n⏭️  Пропуск этапа подготовки данных")
        if not check_data_files():
            print("❌ Ошибка: Подготовленные данные не найдены!")
            print("   Запустите без --skip-prepare или подготовьте данные вручную.")
            return 1
    
    # Этап 2: Обучение моделей
    if not args.skip_train:
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
                '--months', str(args.months),
                '--batch-size', str(args.batch_size),
                '--epochs', str(args.epochs),
                '--patience', str(args.patience)
            ]
            
            if args.encoder_only:
                train_cmd.append('--encoder-only')
            elif args.timeseries_only:
                train_cmd.append('--timeseries-only')
            
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
        print("ЭТАП 3: Бэктестинг")
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

