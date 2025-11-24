"""
Мониторинг производительности стратегии в реальном времени
"""
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from config.monitoring_config import MonitoringConfig


@dataclass
class TradeResult:
    """Результат одной сделки"""
    timestamp: datetime
    profit: float
    confidence: float
    direction: int  # 1 = BUY, -1 = SELL
    entry_price: float
    exit_price: float
    exit_reason: str


class PerformanceMonitor:
    """
    Мониторинг производительности стратегии в реальном времени
    
    Отслеживает метрики и сравнивает с ожидаемыми значениями из бэктеста.
    Автоматически определяет статус системы (NORMAL, WARNING, CRITICAL, STOPPED)
    и применяет соответствующие протоколы защиты.
    """
    
    def __init__(self, 
                 backtest_stats: Dict,
                 config: Optional[MonitoringConfig] = None):
        """
        Args:
            backtest_stats: Ожидаемые метрики из бэктеста
            config: Конфигурация мониторинга
        """
        self.backtest_stats = backtest_stats
        self.config = config if config else MonitoringConfig()
        
        # История торговли
        self.trade_history: List[TradeResult] = []
        self.equity_history: List[Dict] = []
        
        # Скользящие метрики
        self.rolling_metrics = {
            'win_rate': [],
            'profit_factor': [],
            'avg_confidence': [],
            'anomaly_rate': [],
            'sharpe_ratio': []
        }
        
        # Статус системы
        self.status = 'NORMAL'  # NORMAL, WARNING, CRITICAL, STOPPED
        self.status_history: List[Dict] = []
        self.alerts: List[Dict] = []
        
        # Ожидаемые значения из бэктеста
        self.expected_win_rate = backtest_stats.get('win_rate', 0) / 100
        self.expected_profit_factor = backtest_stats.get('profit_factor', 1.0)
        self.expected_avg_profit = backtest_stats.get('avg_profit', 0)
        
    def add_trade(self, 
                  profit: float,
                  confidence: float,
                  timestamp: datetime,
                  direction: int = 0,
                  entry_price: float = 0.0,
                  exit_price: float = 0.0,
                  exit_reason: str = ''):
        """Добавляет результат сделки"""
        trade = TradeResult(
            timestamp=timestamp,
            profit=profit,
            confidence=confidence,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason
        )
        self.trade_history.append(trade)
        
        # Обновляем скользящие метрики
        self._update_rolling_metrics()
        
        # Проверяем на аномалии
        self._check_performance_alerts()
        
    def add_equity(self, equity: float, timestamp: datetime):
        """Добавляет значение equity"""
        self.equity_history.append({
            'timestamp': timestamp,
            'equity': equity
        })
    
    def update_anomaly_rate(self, anomaly_rate: float):
        """Обновляет частоту аномалий входных данных"""
        if len(self.rolling_metrics['anomaly_rate']) == 0:
            self.rolling_metrics['anomaly_rate'] = [anomaly_rate]
        else:
            # Скользящее среднее
            window = min(self.config.window_size, len(self.rolling_metrics['anomaly_rate']) + 1)
            recent_rates = self.rolling_metrics['anomaly_rate'][-window:] + [anomaly_rate]
            avg_rate = np.mean(recent_rates)
            self.rolling_metrics['anomaly_rate'].append(avg_rate)
    
    def update_drift_score(self, drift_score: float):
        """Обновляет оценку дрифта модели"""
        self.current_drift_score = drift_score
    
    def _update_rolling_metrics(self):
        """Обновляет скользящие метрики"""
        if len(self.trade_history) < self.config.window_size:
            return
        
        # Берем последние N сделок
        recent_trades = self.trade_history[-self.config.window_size:]
        
        # Win Rate
        wins = sum(1 for t in recent_trades if t.profit > 0)
        win_rate = wins / len(recent_trades)
        self.rolling_metrics['win_rate'].append(win_rate)
        
        # Profit Factor
        profits = [t.profit for t in recent_trades if t.profit > 0]
        losses = [abs(t.profit) for t in recent_trades if t.profit < 0]
        gross_profit = sum(profits) if profits else 0
        gross_loss = sum(losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        self.rolling_metrics['profit_factor'].append(profit_factor)
        
        # Средняя уверенность
        confidences = [t.confidence for t in recent_trades]
        avg_confidence = np.mean(confidences) if confidences else 0
        self.rolling_metrics['avg_confidence'].append(avg_confidence)
        
        # Sharpe Ratio (упрощенный)
        if len(recent_trades) > 1:
            returns = [t.profit for t in recent_trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / std_return if std_return > 0 else 0
            self.rolling_metrics['sharpe_ratio'].append(sharpe)
    
    def _check_performance_alerts(self):
        """Проверяет на аномалии производительности"""
        if len(self.trade_history) < self.config.window_size:
            return
        
        old_status = self.status
        new_status = 'NORMAL'
        
        # Проверяем пороги для STOPPED (самый критичный)
        if self._check_stopped_thresholds():
            new_status = 'STOPPED'
        # Проверяем пороги для CRITICAL
        elif self._check_critical_thresholds():
            new_status = 'CRITICAL'
        # Проверяем пороги для WARNING
        elif self._check_warning_thresholds():
            new_status = 'WARNING'
        
        # Обновляем статус
        if new_status != old_status:
            self.status = new_status
            self.status_history.append({
                'timestamp': datetime.now(),
                'old_status': old_status,
                'new_status': new_status,
                'reason': self._get_status_change_reason()
            })
    
    def _check_warning_thresholds(self) -> bool:
        """Проверяет пороги для предупреждения"""
        if len(self.rolling_metrics['win_rate']) == 0:
            return False
        
        thresholds = self.config.warning_thresholds
        
        # 1. Проверка Win Rate
        current_wr = self.rolling_metrics['win_rate'][-1]
        if current_wr < self.expected_win_rate * (1 - thresholds['win_rate_drop']):
            self._add_alert('WARNING', 
                f"Win Rate упал: {current_wr:.1%} vs ожидалось {self.expected_win_rate:.1%}")
            return True
        
        # 2. Проверка Profit Factor
        if len(self.rolling_metrics['profit_factor']) > 0:
            current_pf = self.rolling_metrics['profit_factor'][-1]
            if current_pf < self.expected_profit_factor * (1 - thresholds['profit_factor_drop']):
                self._add_alert('WARNING',
                    f"Profit Factor упал: {current_pf:.2f} vs ожидалось {self.expected_profit_factor:.2f}")
                return True
        
        # 3. Проверка средней уверенности
        if len(self.rolling_metrics['avg_confidence']) > 0:
            current_conf = self.rolling_metrics['avg_confidence'][-1]
            # Ожидаемая уверенность из бэктеста (если есть)
            expected_conf = self.backtest_stats.get('avg_confidence', 0.7)
            if current_conf < expected_conf * (1 - thresholds['avg_confidence_drop']):
                self._add_alert('WARNING',
                    f"Средняя уверенность упала: {current_conf:.2f} vs ожидалось {expected_conf:.2f}")
                return True
        
        # 4. Проверка частоты аномалий
        if len(self.rolling_metrics['anomaly_rate']) > 0:
            current_anomaly = self.rolling_metrics['anomaly_rate'][-1]
            if current_anomaly > thresholds['anomaly_rate']:
                self._add_alert('WARNING',
                    f"Высокая частота аномалий: {current_anomaly:.1%} (порог: {thresholds['anomaly_rate']:.1%})")
                return True
        
        # 5. Проверка дрифта модели
        if hasattr(self, 'current_drift_score'):
            if self.current_drift_score > thresholds.get('drift_score', 0.05):
                self._add_alert('WARNING',
                    f"Обнаружен дрифт модели: {self.current_drift_score:.1%} (порог: {thresholds.get('drift_score', 0.05):.1%})")
                return True
        
        return False
    
    def _check_critical_thresholds(self) -> bool:
        """Проверяет пороги для критической ситуации"""
        thresholds = self.config.critical_thresholds
        
        # 1. Проверка серий убытков
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= thresholds['consecutive_losses']:
            self._add_alert('CRITICAL',
                f"Серия убытков: {consecutive_losses} подряд (лимит: {thresholds['consecutive_losses']})")
            return True
        
        # 2. Проверка просадки
        max_drawdown = self._calculate_max_drawdown()
        if max_drawdown > thresholds['max_drawdown']:
            self._add_alert('CRITICAL',
                f"Максимальная просадка: {max_drawdown:.1%} (лимит: {thresholds['max_drawdown']:.1%})")
            return True
        
        # 3. Проверка Win Rate
        if len(self.rolling_metrics['win_rate']) > 0:
            current_wr = self.rolling_metrics['win_rate'][-1]
            if current_wr < self.expected_win_rate * (1 - thresholds['win_rate_drop']):
                self._add_alert('CRITICAL',
                    f"Критическое падение Win Rate: {current_wr:.1%} vs ожидалось {self.expected_win_rate:.1%}")
                return True
        
        # 4. Проверка Profit Factor
        if len(self.rolling_metrics['profit_factor']) > 0:
            current_pf = self.rolling_metrics['profit_factor'][-1]
            if current_pf < self.expected_profit_factor * (1 - thresholds['profit_factor_drop']):
                self._add_alert('CRITICAL',
                    f"Критическое падение Profit Factor: {current_pf:.2f} vs ожидалось {self.expected_profit_factor:.2f}")
                return True
        
        # 5. Проверка дрифта
        if hasattr(self, 'current_drift_score'):
            if self.current_drift_score > thresholds.get('drift_score', 0.10):
                self._add_alert('CRITICAL',
                    f"Критический дрифт модели: {self.current_drift_score:.1%}")
                return True
        
        return False
    
    def _check_stopped_thresholds(self) -> bool:
        """Проверяет пороги для полной остановки"""
        thresholds = self.config.stopped_thresholds
        
        # 1. Серия убытков
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= thresholds['consecutive_losses']:
            self._add_alert('STOPPED',
                f"КРИТИЧЕСКАЯ СЕРИЯ УБЫТКОВ: {consecutive_losses} подряд")
            return True
        
        # 2. Просадка
        max_drawdown = self._calculate_max_drawdown()
        if max_drawdown > thresholds['max_drawdown']:
            self._add_alert('STOPPED',
                f"КРИТИЧЕСКАЯ ПРОСАДКА: {max_drawdown:.1%}")
            return True
        
        # 3. Win Rate
        if len(self.rolling_metrics['win_rate']) > 0:
            current_wr = self.rolling_metrics['win_rate'][-1]
            if current_wr < self.expected_win_rate * (1 - thresholds['win_rate_drop']):
                self._add_alert('STOPPED',
                    f"КРИТИЧЕСКОЕ ПАДЕНИЕ Win Rate: {current_wr:.1%}")
                return True
        
        return False
    
    def _count_consecutive_losses(self) -> int:
        """Считает количество убытков подряд"""
        if not self.trade_history:
            return 0
        
        count = 0
        for trade in reversed(self.trade_history):
            if trade.profit < 0:
                count += 1
            else:
                break
        return count
    
    def _calculate_max_drawdown(self) -> float:
        """Вычисляет максимальную просадку"""
        if len(self.equity_history) < 2:
            return 0.0
        
        equities = [e['equity'] for e in self.equity_history]
        peak = equities[0]
        max_dd = 0.0
        
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _add_alert(self, level: str, message: str):
        """Добавляет алерт"""
        self.alerts.append({
            'timestamp': datetime.now(),
            'level': level,
            'message': message
        })
    
    def _get_status_change_reason(self) -> str:
        """Возвращает причину изменения статуса"""
        if not self.alerts:
            return "Автоматическая проверка"
        return self.alerts[-1]['message']
    
    def should_stop_trading(self) -> bool:
        """Определяет, нужно ли остановить торговлю"""
        return self.status == 'STOPPED'
    
    def should_reduce_position_size(self) -> bool:
        """Определяет, нужно ли снизить размер позиций"""
        return self.status in ['WARNING', 'CRITICAL', 'STOPPED']
    
    def get_position_size_multiplier(self) -> float:
        """Возвращает множитель для размера позиции"""
        return self.config.position_size_multipliers.get(self.status, 1.0)
    
    def get_status_report(self) -> Dict:
        """Возвращает отчет о статусе"""
        current_wr = self.rolling_metrics['win_rate'][-1] if self.rolling_metrics['win_rate'] else None
        current_pf = self.rolling_metrics['profit_factor'][-1] if self.rolling_metrics['profit_factor'] else None
        current_conf = self.rolling_metrics['avg_confidence'][-1] if self.rolling_metrics['avg_confidence'] else None
        
        return {
            'status': self.status,
            'total_trades': len(self.trade_history),
            'current_win_rate': current_wr,
            'expected_win_rate': self.expected_win_rate,
            'current_profit_factor': current_pf,
            'expected_profit_factor': self.expected_profit_factor,
            'current_avg_confidence': current_conf,
            'consecutive_losses': self._count_consecutive_losses(),
            'max_drawdown': self._calculate_max_drawdown(),
            'position_size_multiplier': self.get_position_size_multiplier(),
            'recent_alerts': self.alerts[-5:],  # Последние 5 алертов
            'drift_score': getattr(self, 'current_drift_score', None),
        }
    
    def get_metrics_history(self) -> Dict:
        """Возвращает историю всех метрик для визуализации"""
        return {
            'win_rate': self.rolling_metrics['win_rate'],
            'profit_factor': self.rolling_metrics['profit_factor'],
            'avg_confidence': self.rolling_metrics['avg_confidence'],
            'anomaly_rate': self.rolling_metrics['anomaly_rate'],
            'sharpe_ratio': self.rolling_metrics['sharpe_ratio'],
            'equity_history': self.equity_history,
            'status_history': self.status_history,
        }

