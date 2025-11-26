"""
Скрипт для бэктестинга торговой стратегии
"""
import pandas as pd
from trading.backtester import Backtester
from config.trading_config import TradingConfig
from config.monitoring_config import MonitoringConfig

def main():
    """
    Запуск бэктестинга торговой стратегии
    """
    print("=" * 60)
    print("Бэктестинг торговой стратегии на основе Transformer")
    print("=" * 60)
    
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
    model_type = 'encoder'  # или 'timeseries'
    model_path = f'workspace/models/checkpoints/{model_type}_model.pth'
    # Используем scaler с именем модели (если обучено через train_all_models.py)
    # Или общий scaler (если обучено через train_model.py)
    scaler_path = f'workspace/prepared/scalers/feature_scaler_{model_type}.pkl'
    
    # Проверяем наличие scaler с именем модели, если нет - используем общий
    import os
    if not os.path.exists(scaler_path):
        scaler_path = 'workspace/prepared/scalers/feature_scaler.pkl'
        print(f"   Используется общий scaler: {scaler_path}")
    else:
        print(f"   Используется scaler модели: {scaler_path}")
    
    # Загрузка тестовых данных
    print("\n1. Загрузка данных...")
    test_df = pd.read_csv('workspace/prepared/features/gold_test.csv', index_col=0, parse_dates=True)
    print(f"   Загружено {len(test_df)} свечей")
    
    # Конфигурация мониторинга (можно настроить под свои нужды)
    monitoring_config = MonitoringConfig(
        window_size=50,  # Размер окна для скользящих метрик
        # Пороги можно настроить в config/monitoring_config.py
    )
    
    # Создание бэктестера
    print("\n2. Инициализация бэктестера...")
    backtester = Backtester(
        model_path=model_path,
        scaler_path=scaler_path,
        model_type=model_type,
        trading_config=trading_config,
        monitoring_config=monitoring_config,
        enable_monitoring=True  # Включить мониторинг производительности
    )
    
    # Запуск бэктестинга
    print("\n3. Запуск бэктестинга...")
    results = backtester.backtest(test_df, start_idx=60, save_plots=True)
    
    # Сохранение результатов
    print("\n4. Сохранение результатов...")
    os.makedirs('workspace/results/backtests', exist_ok=True)
    results_df = pd.DataFrame([results])
    results_df.to_csv('workspace/results/backtests/backtest_results.csv', index=False)
    
    # Сохранение истории equity
    if 'equity_history' in results:
        results['equity_history'].to_csv('workspace/results/backtests/equity_history.csv', index=False)
        print("   История equity сохранена: workspace/results/backtests/equity_history.csv")
    
    # Сохранение закрытых позиций
    if backtester.position_manager.closed_positions:
        positions_df = pd.DataFrame(backtester.position_manager.closed_positions)
        positions_df.to_csv('workspace/results/backtests/closed_positions.csv', index=False)
        print("   Закрытые позиции сохранены: workspace/results/backtests/closed_positions.csv")
    
    # Выводим информацию о мониторинге
    if 'performance_monitoring' in results:
        monitor = results['performance_monitoring']
        print(f"\n📊 Мониторинг: Статус = {monitor.get('status', 'NORMAL')}")
        if monitor.get('drift_score') is not None:
            print(f"   Дрифт модели: {monitor['drift_score']:.1%}")
    
    print("\n" + "=" * 60)
    print("Бэктестинг завершен!")
    print("=" * 60)
    print("\n💡 Графики мониторинга сохранены в: workspace/results/monitoring/")
    print("📖 Подробнее: docs/12_PERFORMANCE_MONITORING.md")

if __name__ == '__main__':
    main()

