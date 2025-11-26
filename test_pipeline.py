"""
Тестовый скрипт для проверки полного цикла на 1 месяце данных
"""
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

def test_data_preparation():
    """Тест подготовки данных"""
    print("=" * 60)
    print("ТЕСТ 1: Подготовка данных (1 месяц)")
    print("=" * 60)
    
    try:
        from data.gold_data_prep import GoldDataPreparator
        from data.data_splitter import DataSplitter
        from config.feature_config import FeatureConfig
        
        config = FeatureConfig()
        preparator = GoldDataPreparator(config=config, training_months=1)
        
        # Подготовка данных (1 месяц)
        print("\nЗагрузка данных (1 месяц)...")
        df = preparator.prepare_full_dataset(
            symbol='XAUUSD',
            end_date=None,
            months=1,
            load_ticks=True,
            load_higher_tf=True,
            use_cache=True,  # Использовать кэш
            force_regenerate=False,  # Не принуждать регенерацию
            ask_on_existing=False  # В тестовом режиме не спрашивать
        )
        
        if df.empty:
            print("ОШИБКА: Данные не загружены")
            return False
        
        print(f"✓ Загружено {len(df)} образцов")
        
        # Разделение данных
        print("\nРазделение данных...")
        splitter = DataSplitter(temporal_split=True)
        train_df, val_df, test_df = splitter.split(df, target_column='signal_class')
        
        print(f"✓ Train: {len(train_df)} образцов")
        print(f"✓ Val: {len(val_df)} образцов")
        print(f"✓ Test: {len(test_df)} образцов")
        
        # Сохранение
        train_df.to_csv('workspace/prepared/features/gold_train_test.csv', index=True)
        val_df.to_csv('workspace/prepared/features/gold_val_test.csv', index=True)
        test_df.to_csv('workspace/prepared/features/gold_test_test.csv', index=True)
        
        print("\n✓ Данные сохранены")
        return True
        
    except Exception as e:
        print(f"ОШИБКА при подготовке данных: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_training():
    """Тест обучения модели"""
    print("\n" + "=" * 60)
    print("ТЕСТ 2: Обучение модели")
    print("=" * 60)
    
    try:
        import torch
        from models.model_factory import create_model, get_model_config
        from models.data_loader import create_dataloaders
        from models.trainer import ModelTrainer
        
        # Загрузка данных
        print("\nЗагрузка данных...")
        train_df = pd.read_csv('workspace/prepared/features/gold_train_test.csv', index_col=0, parse_dates=True)
        val_df = pd.read_csv('workspace/prepared/features/gold_val_test.csv', index_col=0, parse_dates=True)
        test_df = pd.read_csv('workspace/prepared/features/gold_test_test.csv', index_col=0, parse_dates=True)
        
        print(f"✓ Загружено данных: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Создание DataLoader'ов
        print("\nСоздание последовательностей...")
        train_loader, val_loader, test_loader, seq_gen = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            sequence_length=60,
            batch_size=16,  # Меньший batch для теста
            target_column='signal_class'
        )
        
        num_features = train_loader.dataset.sequences.shape[2]
        print(f"✓ Количество фичей: {num_features}")
        
        # Создание модели (упрощенная для теста)
        print("\nСоздание модели...")
        model_type = 'encoder'  # Используем простую модель для теста
        
        config = get_model_config(
            model_type=model_type,
            num_features=num_features,
            num_classes=3,
            sequence_length=60,
            d_model=128,  # Уменьшенная размерность для теста
            n_layers=2,   # Меньше слоев для быстрого теста
            n_heads=4,
            dropout=0.1,
            batch_size=16,
            num_epochs=5,  # Всего 5 эпох для теста
            early_stopping_patience=3
        )
        
        model = create_model(config)
        print(f"✓ Модель создана: {sum(p.numel() for p in model.parameters()):,} параметров")
        
        # Обучение
        print("\nОбучение модели (5 эпох для теста)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trainer = ModelTrainer(
            model=model,
            device=device,
            learning_rate=1e-4,
            weight_decay=1e-5,
            scheduler_type='cosine',
            model_config=config  # Передаем конфигурацию для сохранения в checkpoint
        )
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,  # Только 5 эпох для теста
            early_stopping_patience=3,
            checkpoint_path='workspace/models/checkpoints/test_model.pth',
            save_history=True
        )
        
        # Сохранение scaler
        seq_gen.save_scaler('workspace/prepared/scalers/feature_scaler_test.pkl')
        print("\n✓ Обучение завершено")
        return True
        
    except Exception as e:
        print(f"ОШИБКА при обучении модели: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtesting():
    """Тест бэктестинга"""
    print("\n" + "=" * 60)
    print("ТЕСТ 3: Бэктестинг")
    print("=" * 60)
    
    try:
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
        
        # Загрузка тестовых данных
        print("\nЗагрузка данных...")
        test_df = pd.read_csv('workspace/prepared/features/gold_test_test.csv', index_col=0, parse_dates=True)
        print(f"✓ Загружено {len(test_df)} свечей")
        
        # Создание бэктестера
        print("\nИнициализация бэктестера...")
        backtester = Backtester(
            model_path='workspace/models/checkpoints/test_model.pth',
            scaler_path='workspace/prepared/scalers/feature_scaler_test.pkl',
            model_type='encoder',
            trading_config=trading_config
        )
        
        print("✓ Бэктестер создан")
        
        # Запуск бэктестинга
        print("\nЗапуск бэктестинга...")
        results = backtester.backtest(test_df, start_idx=60)
        
        # Сохранение результатов
        results_df = pd.DataFrame([results])
        results_df.to_csv('trading/backtest_results_test.csv', index=False)
        print("\n✓ Результаты сохранены")
        
        return True
        
    except Exception as e:
        print(f"ОШИБКА при бэктестинге: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Главная функция для запуска всех тестов"""
    print("=" * 60)
    print("ТЕСТОВЫЙ ЦИКЛ: Проверка на 1 месяце данных")
    print("=" * 60)
    print("\nЭтот скрипт проверит:")
    print("1. Подготовку данных (1 месяц)")
    print("2. Обучение модели (5 эпох)")
    print("3. Бэктестинг стратегии")
    print("\nВНИМАНИЕ: Убедитесь, что MT5 запущен и подключен!")
    print("=" * 60)
    
    input("\nНажмите Enter для начала тестирования...")
    
    results = []
    
    # Тест 1: Подготовка данных
    result1 = test_data_preparation()
    results.append(("Подготовка данных", result1))
    
    if not result1:
        print("\n❌ Тест подготовки данных провален. Остановка.")
        return
    
    # Тест 2: Обучение модели
    result2 = test_model_training()
    results.append(("Обучение модели", result2))
    
    if not result2:
        print("\n❌ Тест обучения модели провален. Остановка.")
        return
    
    # Тест 3: Бэктестинг
    result3 = test_backtesting()
    results.append(("Бэктестинг", result3))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ ПРОЙДЕН" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("\nСледующие шаги:")
        print("1. Запустите полный цикл на 6 месяцах данных:")
        print("   python prepare_gold_data.py")
        print("   python train_model.py")
        print("   python backtest_strategy.py")
        print("2. Проанализируйте результаты бэктестинга")
        print("3. Оптимизируйте параметры торговли")
    else:
        print("\n⚠️ Некоторые тесты провалены. Проверьте ошибки выше.")
    
    print("=" * 60)

if __name__ == '__main__':
    main()

