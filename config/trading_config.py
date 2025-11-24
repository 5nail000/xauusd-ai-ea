"""
Конфигурация торговых параметров
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradingConfig:
    """Конфигурация параметров торговли"""
    
    # Размер позиции
    base_lot_size: float = 0.1  # Базовый размер лота
    lot_multiplier: float = 1.0  # Множитель для надежных сигналов
    
    # Тейк-профит и стоп-лосс (в пунктах)
    take_profit_min: float = 50.0  # Минимальный тейк-профит
    take_profit_max: float = 130.0  # Максимальный тейк-профит
    stop_loss: float = 100.0  # Стоп-лосс
    
    # Трейлинг стоп
    use_trailing_stop: bool = True  # Использовать трейлинг стоп
    trailing_start: float = 30.0  # Активация трейлинга после N пунктов прибыли
    trailing_step: float = 20.0  # Шаг трейлинга в пунктах
    
    # Частичное закрытие
    use_partial_close: bool = True  # Использовать частичное закрытие
    partial_close_at: float = 60.0  # Частичное закрытие при достижении N пунктов
    partial_close_ratio: float = 0.5  # Доля позиции для закрытия (0.5 = 50%)
    
    # Множитель для надежных сигналов
    use_signal_confidence: bool = True  # Использовать уверенность модели
    confidence_threshold: float = 0.8  # Порог уверенности для увеличения лота
    confidence_multiplier: float = 1.5  # Множитель лота для надежных сигналов
    
    # Комиссия и спред
    commission_per_lot: float = 0.0  # Комиссия за лот (если есть)
    spread_pips: float = 2.0  # Спред в пунктах (для золота обычно 2-3)
    
    # Управление позициями
    max_open_positions: int = 1  # Максимальное количество открытых позиций
    allow_opposite_positions: bool = False  # Разрешить противоположные позиции
    
    def calculate_lot_size(self, signal_confidence: Optional[float] = None) -> float:
        """
        Вычисляет размер лота на основе уверенности сигнала
        
        Args:
            signal_confidence: Уверенность модели (0-1) - пока не используется, всегда базовый лот
        
        Returns:
            Размер лота
        """
        # Пока всегда используем базовый лот
        # Уверенность сохраняется для будущего использования в стратегии
        lot_size = self.base_lot_size
        
        # TODO: В будущем можно использовать уверенность для управления размером лота
        # if self.use_signal_confidence and signal_confidence is not None:
        #     if signal_confidence >= self.confidence_threshold:
        #         lot_size *= self.confidence_multiplier
        
        return lot_size
    
    def calculate_take_profit(self, entry_price: float, direction: int, 
                             signal_confidence: Optional[float] = None) -> float:
        """
        Вычисляет тейк-профит
        
        Args:
            entry_price: Цена входа
            direction: Направление (1 для покупки, -1 для продажи)
            signal_confidence: Уверенность сигнала
        
        Returns:
            Цена тейк-профита
        """
        # Базовый тейк-профит
        tp_pips = self.take_profit_min
        
        # Увеличиваем TP для надежных сигналов
        if self.use_signal_confidence and signal_confidence is not None:
            if signal_confidence >= self.confidence_threshold:
                # Линейная интерполяция между min и max
                confidence_factor = (signal_confidence - self.confidence_threshold) / (1.0 - self.confidence_threshold)
                tp_pips = self.take_profit_min + (self.take_profit_max - self.take_profit_min) * confidence_factor
            else:
                tp_pips = self.take_profit_min
        
        # Конвертируем пункты в цену (для золота 1 пункт = 0.01)
        tp_price = entry_price + (direction * tp_pips * 0.01)
        
        return tp_price
    
    def calculate_stop_loss(self, entry_price: float, direction: int) -> float:
        """
        Вычисляет стоп-лосс
        
        Args:
            entry_price: Цена входа
            direction: Направление (1 для покупки, -1 для продажи)
        
        Returns:
            Цена стоп-лосса
        """
        sl_price = entry_price - (direction * self.stop_loss * 0.01)
        return sl_price

# Конфигурация по умолчанию
default_trading_config = TradingConfig()

