"""
Модуль управления позициями в бэктестинге
"""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import numpy as np

@dataclass
class Position:
    """Класс для представления торговой позиции"""
    entry_time: datetime
    entry_price: float
    direction: int  # 1 для покупки, -1 для продажи
    lot_size: float
    take_profit: float
    stop_loss: float
    trailing_stop: Optional[float] = None
    partial_closed: bool = False
    signal_confidence: Optional[float] = None
    
    def update_trailing_stop(self, current_price: float, trailing_start: float, trailing_step: float):
        """
        Обновляет трейлинг стоп
        
        Args:
            current_price: Текущая цена
            trailing_start: Активация трейлинга после N пунктов прибыли
            trailing_step: Шаг трейлинга в пунктах
        """
        # Вычисляем текущую прибыль в пунктах
        profit_pips = abs(current_price - self.entry_price) * 100
        
        if profit_pips >= trailing_start:
            # Вычисляем новый уровень трейлинга
            if self.direction == 1:  # Покупка
                new_trailing = current_price - (trailing_step * 0.01)
                if self.trailing_stop is None or new_trailing > self.trailing_stop:
                    self.trailing_stop = new_trailing
            else:  # Продажа
                new_trailing = current_price + (trailing_step * 0.01)
                if self.trailing_stop is None or new_trailing < self.trailing_stop:
                    self.trailing_stop = new_trailing
    
    def check_exit(self, current_price: float, spread: float = 0.0) -> tuple:
        """
        Проверяет условия выхода из позиции
        
        Args:
            current_price: Текущая цена
            spread: Спред в пунктах
        
        Returns:
            Tuple (should_exit, exit_reason, exit_price)
            - should_exit: True если нужно выйти
            - exit_reason: Причина выхода ('tp', 'sl', 'trailing')
            - exit_price: Цена выхода
        """
        # Проверка тейк-профита
        if self.direction == 1:  # Покупка
            if current_price >= self.take_profit:
                return True, 'tp', self.take_profit
        else:  # Продажа
            if current_price <= self.take_profit:
                return True, 'tp', self.take_profit
        
        # Проверка стоп-лосса
        if self.direction == 1:  # Покупка
            if current_price <= self.stop_loss:
                return True, 'sl', self.stop_loss
        else:  # Продажа
            if current_price >= self.stop_loss:
                return True, 'sl', self.stop_loss
        
        # Проверка трейлинг стопа
        if self.trailing_stop is not None:
            if self.direction == 1:  # Покупка
                if current_price <= self.trailing_stop:
                    return True, 'trailing', self.trailing_stop
            else:  # Продажа
                if current_price >= self.trailing_stop:
                    return True, 'trailing', self.trailing_stop
        
        return False, None, None
    
    def check_partial_close(self, current_price: float, partial_close_at: float) -> bool:
        """
        Проверяет условие частичного закрытия
        
        Args:
            current_price: Текущая цена
            partial_close_at: Уровень для частичного закрытия в пунктах
        
        Returns:
            True если нужно частично закрыть
        """
        if self.partial_closed:
            return False
        
        profit_pips = abs(current_price - self.entry_price) * 100
        
        if profit_pips >= partial_close_at:
            return True
        
        return False
    
    def calculate_profit(self, exit_price: float, spread: float = 0.0, 
                        commission: float = 0.0) -> float:
        """
        Вычисляет прибыль/убыток
        
        Args:
            exit_price: Цена выхода
            spread: Спред в пунктах
            commission: Комиссия за лот
        
        Returns:
            Прибыль/убыток в валюте депозита
        """
        # Для золота: 1 пункт = 0.01 в цене
        # При лоте 0.1: 1 пункт = $0.1 прибыли
        # Формула: (exit_price - entry_price) * direction * lot_size * 100
        price_diff = (exit_price - self.entry_price) * self.direction
        profit = price_diff * 100 * self.lot_size * 10  # 10 = конвертация для золота (1 лот = 100 унций)
        
        # Учитываем спред и комиссию
        spread_cost = spread * self.lot_size * 10
        commission_cost = commission * self.lot_size
        
        net_profit = profit - spread_cost - commission_cost
        
        return net_profit

class PositionManager:
    """Менеджер позиций для бэктестинга"""
    
    def __init__(self, config):
        """
        Args:
            config: TradingConfig
        """
        self.config = config
        self.positions = []
        self.closed_positions = []
        self.equity = []
        self.balance = 10000.0  # Начальный баланс
        self.current_equity = self.balance
    
    def can_open_position(self, direction: int) -> bool:
        """
        Проверяет, можно ли открыть новую позицию
        
        Args:
            direction: Направление (1 для покупки, -1 для продажи)
        
        Returns:
            True если можно открыть
        """
        # Проверка максимального количества позиций
        if len(self.positions) >= self.config.max_open_positions:
            return False
        
        # Проверка противоположных позиций
        if not self.config.allow_opposite_positions:
            for pos in self.positions:
                if pos.direction != direction:
                    return False
        
        return True
    
    def open_position(self, entry_time: datetime, entry_price: float, 
                     direction: int, signal_confidence: Optional[float] = None) -> Optional[Position]:
        """
        Открывает новую позицию
        
        Args:
            entry_time: Время входа
            entry_price: Цена входа
            direction: Направление (1 для покупки, -1 для продажи)
            signal_confidence: Уверенность сигнала
        
        Returns:
            Position или None если не удалось открыть
        """
        if not self.can_open_position(direction):
            return None
        
        # Вычисляем размер лота
        lot_size = self.config.calculate_lot_size(signal_confidence)
        
        # Вычисляем тейк-профит и стоп-лосс
        take_profit = self.config.calculate_take_profit(entry_price, direction, signal_confidence)
        stop_loss = self.config.calculate_stop_loss(entry_price, direction)
        
        # Создаем позицию
        position = Position(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=direction,
            lot_size=lot_size,
            take_profit=take_profit,
            stop_loss=stop_loss,
            signal_confidence=signal_confidence
        )
        
        self.positions.append(position)
        return position
    
    def update_positions(self, current_time: datetime, current_price: float):
        """
        Обновляет все открытые позиции
        
        Args:
            current_time: Текущее время
            current_price: Текущая цена
        """
        positions_to_close = []
        
        for position in self.positions:
            # Обновляем трейлинг стоп
            if self.config.use_trailing_stop:
                position.update_trailing_stop(
                    current_price,
                    self.config.trailing_start,
                    self.config.trailing_step
                )
            
            # Проверяем частичное закрытие
            if self.config.use_partial_close:
                if position.check_partial_close(current_price, self.config.partial_close_at):
                    # Частично закрываем позицию
                    partial_lot = position.lot_size * self.config.partial_close_ratio
                    position.lot_size -= partial_lot
                    position.partial_closed = True
                    
                    # Вычисляем прибыль от частичного закрытия
                    profit = (current_price - position.entry_price) * position.direction
                    profit = profit * 100 * partial_lot * 10
                    profit -= self.config.spread_pips * partial_lot * 10
                    profit -= self.config.commission_per_lot * partial_lot
                    
                    self.balance += profit
                    self.current_equity = self.balance
            
            # Проверяем условия выхода
            should_exit, exit_reason, exit_price = position.check_exit(
                current_price,
                self.config.spread_pips
            )
            
            if should_exit:
                positions_to_close.append((position, exit_reason, exit_price))
        
        # Закрываем позиции
        for position, exit_reason, exit_price in positions_to_close:
            self.close_position(position, current_time, exit_price, exit_reason)
    
    def close_position(self, position: Position, exit_time: datetime, 
                      exit_price: float, exit_reason: str):
        """
        Закрывает позицию
        
        Args:
            position: Позиция для закрытия
            exit_time: Время выхода
            exit_price: Цена выхода
            exit_reason: Причина выхода
        """
        # Вычисляем прибыль
        profit = position.calculate_profit(
            exit_price,
            self.config.spread_pips,
            self.config.commission_per_lot
        )
        
        # Обновляем баланс
        self.balance += profit
        self.current_equity = self.balance
        
        # Сохраняем закрытую позицию
        closed_pos = {
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'direction': position.direction,
            'lot_size': position.lot_size,
            'profit': profit,
            'exit_reason': exit_reason,
            'signal_confidence': position.signal_confidence
        }
        
        self.closed_positions.append(closed_pos)
        
        # Удаляем из открытых позиций
        self.positions.remove(position)
    
    def get_statistics(self) -> dict:
        """
        Возвращает статистику торговли
        
        Returns:
            Словарь со статистикой
        """
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'avg_profit': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0
            }
        
        profits = [pos['profit'] for pos in self.closed_positions]
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_profit = sum(profits)
        win_rate = len(winning_trades) / len(profits) * 100
        
        avg_profit = np.mean(profits) if profits else 0.0
        max_profit = max(profits) if profits else 0.0
        max_loss = min(profits) if profits else 0.0
        
        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Статистика по причинам выхода
        exit_reasons = {}
        for pos in self.closed_positions:
            reason = pos['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'total_trades': len(self.closed_positions),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'exit_reasons': exit_reasons,
            'final_balance': self.balance,
            'return_pct': (self.balance - 10000.0) / 10000.0 * 100
        }

