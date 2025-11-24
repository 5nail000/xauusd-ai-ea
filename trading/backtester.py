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
import warnings

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
        
        # Инициализируем генератор последовательностей
        self.sequence_generator = SequenceGenerator(sequence_length=60)
        self.scaler_metadata = {}  # Будет заполнено при загрузке в backtest()
        self.feature_stats = None  # Будет загружено позже
        self.anomaly_threshold = 3.0  # 3 стандартных отклонения
        
        # Статистика аномалий
        self.anomaly_stats = {
            'total_checks': 0,
            'anomalies_detected': 0,
            'signals_skipped': 0,
            'confidence_reduced': 0
        }
        
        # Загружаем модель
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        self.model.eval()
        
        # Менеджер позиций
        self.position_manager = PositionManager(self.trading_config)
    
    def _load_model(self) -> torch.nn.Module:
        """Загружает обученную модель"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Определяем num_features из scaler (без валидации, т.к. DataFrame еще не загружен)
        self.sequence_generator.load_scaler(self.scaler_path, validate_features=False)
        num_features = len(self.sequence_generator.feature_columns) if self.sequence_generator.feature_columns else 100
        
        # Проверяем, есть ли сохраненная конфигурация в checkpoint
        if 'model_config' in checkpoint:
            # Используем сохраненную конфигурацию
            config_dict = checkpoint['model_config']
            # Обновляем num_features из scaler (может отличаться)
            config_dict['num_features'] = num_features
            # Создаем TransformerConfig из словаря
            config = TransformerConfig(**config_dict)
            print(f"[Backtester] Загружена сохраненная конфигурация модели: d_model={config.d_model}, n_layers={config.n_layers}")
        else:
            # Используем дефолтную конфигурацию (для старых checkpoint'ов)
            warnings.warn(
                f"Конфигурация модели не найдена в checkpoint. Используются дефолтные параметры. "
                f"Рекомендуется переобучить модель для сохранения конфигурации.",
                UserWarning
            )
            config = get_model_config(
                model_type=self.model_type,
                num_features=num_features,
                num_classes=3
            )
        
        model = create_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def _load_feature_stats(self, scaler_path: str) -> Optional[Dict]:
        """Загружает статистику фичей для мониторинга аномалий"""
        import pickle
        try:
            with open(scaler_path, 'rb') as f:
                data = pickle.load(f)
                return data.get('feature_stats', None)
        except Exception as e:
            print(f"[Backtester] ⚠ Не удалось загрузить статистику фичей: {e}")
            return None
    
    def _check_anomalies(self, sequence: np.ndarray) -> tuple:
        """
        Проверяет наличие аномалий в последовательности
        
        Args:
            sequence: Массив последовательности [seq_len, n_features]
        
        Returns:
            Tuple (has_anomaly, anomaly_ratio, max_deviation)
            - has_anomaly: есть ли аномалии
            - anomaly_ratio: доля аномальных значений (0-1)
            - max_deviation: максимальное отклонение в сигмах
        """
        if self.feature_stats is None or len(self.feature_stats) == 0:
            return False, 0.0, 0.0
        
        if self.sequence_generator.feature_columns is None:
            return False, 0.0, 0.0
        
        # Берем последний временной шаг последовательности
        last_step = sequence[-1]  # [n_features]
        
        anomaly_count = 0
        max_deviation = 0.0
        
        for i, feature_name in enumerate(self.sequence_generator.feature_columns):
            if i >= len(last_step):
                continue
            
            if feature_name not in self.feature_stats:
                continue
            
            feature_value = last_step[i]
            stats = self.feature_stats[feature_name]
            mean = stats['mean']
            std = stats['std']
            
            if std > 0:
                # Вычисляем отклонение в сигмах
                deviation = abs(feature_value - mean) / std
                max_deviation = max(max_deviation, deviation)
                
                if deviation > self.anomaly_threshold:
                    anomaly_count += 1
        
        total_features = len(self.sequence_generator.feature_columns)
        anomaly_ratio = anomaly_count / total_features if total_features > 0 else 0.0
        has_anomaly = anomaly_ratio > 0.1  # Аномалия если >10% фичей выходят за 3σ
        
        return has_anomaly, anomaly_ratio, max_deviation
    
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
            
            # Проверка аномалий
            self.anomaly_stats['total_checks'] += 1
            has_anomaly, anomaly_ratio, max_deviation = self._check_anomalies(sequence)
            
            anomaly_penalty = 0.0
            if has_anomaly:
                self.anomaly_stats['anomalies_detected'] += 1
                # Снижаем уверенность пропорционально аномалии
                anomaly_penalty = min(anomaly_ratio * 0.5, 0.7)  # Максимум 70% снижения
                
                # Если аномалия очень сильная (>50% фичей), пропускаем сигнал
                if anomaly_ratio > 0.5:
                    self.anomaly_stats['signals_skipped'] += 1
                    return None
                
                # Иначе снижаем уверенность
                self.anomaly_stats['confidence_reduced'] += 1
            
            # Предсказание
            predictions, probabilities = self.predict(sequence_tensor)
            predicted_class = predictions[0]
            confidence = probabilities[0][predicted_class]
            
            # Применяем штраф за аномалии
            if has_anomaly:
                confidence = confidence * (1.0 - anomaly_penalty)
            
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
    
    def backtest(self, df: pd.DataFrame, start_idx: int = 60, 
                 validate_features: bool = True) -> Dict:
        """
        Запускает бэктестинг
        
        Args:
            df: DataFrame с данными для бэктестинга
            start_idx: Начальный индекс (минимально 60 для последовательности)
            validate_features: Проверять ли соответствие фичей (по умолчанию True)
        
        Returns:
            Словарь с результатами бэктестинга
        """
        # Загружаем scaler с валидацией фичей
        if not self.sequence_generator.is_fitted:
            print("Загрузка scaler с валидацией фичей...")
            self.scaler_metadata = self.sequence_generator.load_scaler(
                self.scaler_path, 
                validate_features=validate_features,
                df=df
            )
            
            # Загружаем статистику для мониторинга аномалий
            self.feature_stats = self._load_feature_stats(self.scaler_path)
            
            # Выводим информацию о загруженных метаданных
            if self.scaler_metadata:
                print(f"✓ Загружены метаданные:")
                if 'training_months' in self.scaler_metadata:
                    print(f"  Месяцев данных при обучении: {self.scaler_metadata['training_months']}")
                if 'num_features' in self.scaler_metadata:
                    print(f"  Количество фичей: {self.scaler_metadata['num_features']}")
                if 'preparation_config' in self.scaler_metadata:
                    prep_config = self.scaler_metadata['preparation_config']
                    if prep_config.get('remove_correlated_features'):
                        print(f"  Удаление коррелированных фичей: Да (порог: {prep_config.get('correlation_threshold', 0.95)})")
        
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
            
            # Обновляем открытые позиции и логируем закрытия
            closed_before = len(self.position_manager.closed_positions)
            self.position_manager.update_positions(current_time, current_price)
            closed_after = len(self.position_manager.closed_positions)
            
            # Логируем закрытые позиции
            if closed_after > closed_before:
                for pos in self.position_manager.closed_positions[closed_before:]:
                    profit_sign = "+" if pos['profit'] >= 0 else ""
                    print(f"{pos['exit_time']}: Закрыта позиция {('BUY' if pos['direction'] == 1 else 'SELL')} "
                          f"по цене {pos['exit_price']:.2f}, прибыль {profit_sign}${pos['profit']:.2f}, "
                          f"причина: {pos['exit_reason']} "
                          f"(вход: {pos['entry_price']:.2f}, длительность: {pos['exit_time'] - pos['entry_time']})")
            
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
                              f"уверенность {confidence:.2f}, TP={position.take_profit:.2f}, SL={position.stop_loss:.2f}")
            
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
        print(f"\n{'='*60}")
        print(f"{'СТАТИСТИКА ТОРГОВЛИ':^60}")
        print(f"{'='*60}")
        
        print(f"\n📊 Общая статистика:")
        print(f"  Всего сделок: {stats['total_trades']}")
        print(f"  Прибыльных: {stats['winning_trades']} ({stats['winning_trades']/stats['total_trades']*100:.1f}%)" if stats['total_trades'] > 0 else "  Прибыльных: 0")
        print(f"  Убыточных: {stats['losing_trades']} ({stats['losing_trades']/stats['total_trades']*100:.1f}%)" if stats['total_trades'] > 0 else "  Убыточных: 0")
        print(f"  Win Rate: {stats['win_rate']:.2f}%")
        
        print(f"\n💰 Финансовые результаты:")
        profit_sign = "+" if stats['total_profit'] >= 0 else ""
        print(f"  Общая прибыль/убыток: {profit_sign}${stats['total_profit']:.2f}")
        print(f"  Средняя прибыль на сделку: ${stats['avg_profit']:.2f}")
        print(f"  Максимальная прибыль: ${stats['max_profit']:.2f}")
        print(f"  Максимальный убыток: ${stats['max_loss']:.2f}")
        
        print(f"\n📈 Анализ эффективности:")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Валовой доход: ${stats['gross_profit']:.2f}")
        print(f"  Валовой убыток: ${stats['gross_loss']:.2f}")
        
        print(f"\n💵 Баланс:")
        print(f"  Начальный баланс: $10,000.00")
        print(f"  Финальный баланс: ${stats['final_balance']:.2f}")
        return_sign = "+" if stats['return_pct'] >= 0 else ""
        print(f"  Доходность: {return_sign}{stats['return_pct']:.2f}%")
        
        if 'exit_reasons' in stats and stats['exit_reasons']:
            print(f"\n🚪 Причины выхода из позиций:")
            reason_names = {
                'tp': 'Take Profit',
                'sl': 'Stop Loss',
                'trailing': 'Trailing Stop',
                'end_of_data': 'Конец данных'
            }
            for reason, count in stats['exit_reasons'].items():
                reason_name = reason_names.get(reason, reason)
                print(f"  {reason_name}: {count} ({count/stats['total_trades']*100:.1f}%)" if stats['total_trades'] > 0 else f"  {reason_name}: {count}")
        
        # Статистика аномалий
        if self.anomaly_stats['total_checks'] > 0:
            print(f"\n⚠️  Мониторинг аномалий:")
            print(f"  Всего проверок: {self.anomaly_stats['total_checks']}")
            anomaly_pct = self.anomaly_stats['anomalies_detected'] / self.anomaly_stats['total_checks'] * 100
            print(f"  Обнаружено аномалий: {self.anomaly_stats['anomalies_detected']} ({anomaly_pct:.1f}%)")
            print(f"  Сигналов пропущено: {self.anomaly_stats['signals_skipped']}")
            print(f"  Уверенность снижена: {self.anomaly_stats['confidence_reduced']} раз")
        
        print(f"\n{'='*60}")

