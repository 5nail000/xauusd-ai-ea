"""
Модуль бэктестинга торговой стратегии
"""
import pandas as pd
import numpy as np
import torch
from typing import Optional, Dict
from datetime import datetime
from models.model_factory import create_model, get_model_config
from models.data_loader import SequenceGenerator
from trading.position_manager import PositionManager
from config.trading_config import TradingConfig
from config.model_config import TransformerConfig

class Backtester:
    """
    Класс для бэктестинга торговой стратегии на основе Transformer модели
    """
    
    def __init__(self,
                 model_path: str,
                 scaler_path: str,
                 model_type: str = 'encoder',
                 trading_config: Optional[TradingConfig] = None):
        """
        Args:
            model_path: Путь к обученной модели
            scaler_path: Путь к scaler
            model_type: Тип модели ('encoder' или 'timeseries')
            trading_config: Конфигурация торговли
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model_type = model_type
        self.trading_config = trading_config if trading_config else TradingConfig()
        
        # Загружаем scaler сначала (нужен для определения num_features)
        self.sequence_generator = SequenceGenerator(sequence_length=60)
        self.sequence_generator.load_scaler(scaler_path)
        
        # Загружаем модель
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.model.eval()
        
        # Менеджер позиций
        self.position_manager = PositionManager(self.trading_config)
    
    def _load_model(self) -> torch.nn.Module:
        """Загружает обученную модель"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Определяем num_features из scaler
        self.sequence_generator.load_scaler(self.scaler_path)
        num_features = len(self.sequence_generator.feature_columns) if self.sequence_generator.feature_columns else 100
        
        # Определяем конфигурацию модели
        config = get_model_config(
            model_type=self.model_type,
            num_features=num_features,
            num_classes=3
        )
        
        model = create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def predict(self, sequences: torch.Tensor) -> tuple:
        """
        Делает предсказание модели
        
        Args:
            sequences: Тензор последовательностей [batch, seq_len, features]
        
        Returns:
            Tuple (predictions, confidences)
            - predictions: предсказанные классы
            - confidences: уверенность модели (softmax вероятности)
        """
        with torch.no_grad():
            sequences = sequences.to(self.device)
            outputs = self.model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def get_signal(self, df: pd.DataFrame, idx: int) -> Optional[tuple]:
        """
        Получает торговый сигнал для текущего момента
        
        Args:
            df: DataFrame с данными
            idx: Индекс текущего момента
        
        Returns:
            Tuple (direction, confidence) или None
            - direction: 1 для покупки, -1 для продажи, 0 для удержания
            - confidence: уверенность модели
        """
        if idx < 60:  # Нужно минимум 60 свечей для последовательности
            return None
        
        # Создаем последовательность
        try:
            # Берем данные до текущего момента включительно
            df_subset = df.iloc[:idx+1].copy()
            
            # Создаем последовательности
            sequences, _ = self.sequence_generator.create_sequences(df_subset)
            
            if len(sequences) == 0:
                return None
            
            # Берем последнюю последовательность
            sequence = sequences[-1:]
            sequence_tensor = torch.FloatTensor(sequence)
            
            # Предсказание
            predictions, probabilities = self.predict(sequence_tensor)
            predicted_class = predictions[0]
            confidence = probabilities[0][predicted_class]
            
            # Преобразуем класс в направление
            # 0 = неопределенность (удержание)
            # 1 = пробой (покупка для восходящего пробоя, продажа для нисходящего)
            # 2 = отскок (продажа для восходящего отскока, покупка для нисходящего)
            
            if predicted_class == 0:  # Неопределенность
                return None
            
            # Для пробоя и отскока определяем направление по тренду
            # Упрощенная логика: смотрим на последние цены
            if idx >= 10:
                recent_prices = df['close'].iloc[idx-10:idx+1].values
                trend = 1 if recent_prices[-1] > recent_prices[0] else -1
            else:
                trend = 1  # По умолчанию восходящий тренд
            
            if predicted_class == 1:  # Пробой
                direction = trend  # Пробой по направлению тренда
            else:  # Отскок (класс 2)
                direction = -trend  # Отскок против тренда
            
            return direction, float(confidence)
        
        except Exception as e:
            # print(f"Ошибка при получении сигнала на индексе {idx}: {e}")
            return None
    
    def backtest(self, df: pd.DataFrame, start_idx: int = 60) -> Dict:
        """
        Запускает бэктестинг
        
        Args:
            df: DataFrame с данными для бэктестинга
            start_idx: Начальный индекс (минимально 60 для последовательности)
        
        Returns:
            Словарь с результатами бэктестинга
        """
        print("=" * 60)
        print("Запуск бэктестинга")
        print("=" * 60)
        print(f"Период: {df.index[start_idx]} - {df.index[-1]}")
        print(f"Количество свечей: {len(df) - start_idx}")
        print(f"Начальный баланс: ${self.position_manager.balance:.2f}")
        print("-" * 60)
        
        equity_history = []
        
        for i in range(start_idx, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Обновляем открытые позиции
            self.position_manager.update_positions(current_time, current_price)
            
            # Получаем сигнал
            signal = self.get_signal(df, i)
            
            # Открываем новую позицию если есть сигнал
            if signal is not None:
                direction, confidence = signal
                
                if self.position_manager.can_open_position(direction):
                    position = self.position_manager.open_position(
                        entry_time=current_time,
                        entry_price=current_price,
                        direction=direction,
                        signal_confidence=confidence
                    )
                    
                    if position:
                        print(f"{current_time}: Открыта позиция {('BUY' if direction == 1 else 'SELL')} "
                              f"по цене {current_price:.2f}, лот {position.lot_size:.2f}, "
                              f"уверенность {confidence:.2f}")
            
            # Записываем equity
            equity = self.position_manager.balance
            # Добавляем незакрытую прибыль открытых позиций
            for pos in self.position_manager.positions:
                unrealized_profit = pos.calculate_profit(current_price)
                equity += unrealized_profit
            
            equity_history.append({
                'time': current_time,
                'equity': equity,
                'balance': self.position_manager.balance
            })
        
        # Закрываем все открытые позиции в конце
        final_price = df['close'].iloc[-1]
        for position in self.position_manager.positions[:]:
            self.position_manager.close_position(
                position,
                df.index[-1],
                final_price,
                'end_of_data'
            )
        
        # Получаем статистику
        stats = self.position_manager.get_statistics()
        
        # Добавляем историю equity
        stats['equity_history'] = pd.DataFrame(equity_history)
        
        print("\n" + "=" * 60)
        print("Результаты бэктестинга")
        print("=" * 60)
        self._print_statistics(stats)
        
        return stats
    
    def _print_statistics(self, stats: Dict):
        """Выводит статистику бэктестинга"""
        print(f"Всего сделок: {stats['total_trades']}")
        print(f"Прибыльных: {stats['winning_trades']}")
        print(f"Убыточных: {stats['losing_trades']}")
        print(f"Win Rate: {stats['win_rate']:.2f}%")
        print(f"\nОбщая прибыль: ${stats['total_profit']:.2f}")
        print(f"Средняя прибыль: ${stats['avg_profit']:.2f}")
        print(f"Максимальная прибыль: ${stats['max_profit']:.2f}")
        print(f"Максимальный убыток: ${stats['max_loss']:.2f}")
        print(f"\nProfit Factor: {stats['profit_factor']:.2f}")
        print(f"Валовой доход: ${stats['gross_profit']:.2f}")
        print(f"Валовой убыток: ${stats['gross_loss']:.2f}")
        print(f"\nФинальный баланс: ${stats['final_balance']:.2f}")
        print(f"Доходность: {stats['return_pct']:.2f}%")
        
        if 'exit_reasons' in stats:
            print(f"\nПричины выхода:")
            for reason, count in stats['exit_reasons'].items():
                print(f"  {reason}: {count}")

