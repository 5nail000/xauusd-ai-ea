"""
Скрипт для бэктестинга торговой стратегии
"""
import pandas as pd
from trading.backtester import Backtester
from config.trading_config import TradingConfig

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
    model_path = f'models/checkpoints/{model_type}_model.pth'
    scaler_path = 'models/feature_scaler.pkl'
    
    # Загрузка тестовых данных
    print("\n1. Загрузка данных...")
    test_df = pd.read_csv('data/gold_test.csv', index_col=0, parse_dates=True)
    print(f"   Загружено {len(test_df)} свечей")
    
    # Создание бэктестера
    print("\n2. Инициализация бэктестера...")
    backtester = Backtester(
        model_path=model_path,
        scaler_path=scaler_path,
        model_type=model_type,
        trading_config=trading_config
    )
    
    # Запуск бэктестинга
    print("\n3. Запуск бэктестинга...")
    results = backtester.backtest(test_df, start_idx=60)
    
    # Сохранение результатов
    print("\n4. Сохранение результатов...")
    results_df = pd.DataFrame([results])
    results_df.to_csv('trading/backtest_results.csv', index=False)
    
    # Сохранение истории equity
    if 'equity_history' in results:
        results['equity_history'].to_csv('trading/equity_history.csv', index=False)
        print("   История equity сохранена: trading/equity_history.csv")
    
    # Сохранение закрытых позиций
    if backtester.position_manager.closed_positions:
        positions_df = pd.DataFrame(backtester.position_manager.closed_positions)
        positions_df.to_csv('trading/closed_positions.csv', index=False)
        print("   Закрытые позиции сохранены: trading/closed_positions.csv")
    
    print("\n" + "=" * 60)
    print("Бэктестинг завершен!")
    print("=" * 60)

if __name__ == '__main__':
    main()

