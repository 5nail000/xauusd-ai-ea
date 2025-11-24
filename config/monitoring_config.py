"""
Конфигурация системы мониторинга производительности и защиты от деградации модели
"""
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class MonitoringConfig:
    """Конфигурация мониторинга производительности"""
    
    # Размер окна для скользящих метрик
    window_size: int = 50  # Количество последних сделок для анализа
    
    # Пороги для предупреждения (WARNING)
    warning_thresholds: Dict = None
    
    # Пороги для критической ситуации (CRITICAL)
    critical_thresholds: Dict = None
    
    # Пороги для полной остановки (STOPPED)
    stopped_thresholds: Dict = None
    
    # Множители размера позиций для разных статусов
    position_size_multipliers: Dict = None
    
    # Параметры детектора дрифта
    drift_detection: Dict = None
    
    def __post_init__(self):
        """Инициализация значений по умолчанию"""
        if self.warning_thresholds is None:
            self.warning_thresholds = {
                'win_rate_drop': 0.15,  # Падение Win Rate на 15% от ожидаемого
                'profit_factor_drop': 0.30,  # Падение Profit Factor на 30%
                'avg_confidence_drop': 0.20,  # Падение средней уверенности на 20%
                'anomaly_rate': 0.15,  # >15% аномалий входных данных
                'drift_score': 0.05,  # Дрифт модели > 5%
            }
        
        if self.critical_thresholds is None:
            self.critical_thresholds = {
                'consecutive_losses': 5,  # 5 убытков подряд
                'max_drawdown': 0.20,  # Просадка > 20%
                'win_rate_drop': 0.30,  # Падение Win Rate на 30%
                'profit_factor_drop': 0.50,  # Падение Profit Factor на 50%
                'drift_score': 0.10,  # Дрифт модели > 10%
                'anomaly_rate': 0.25,  # >25% аномалий
            }
        
        if self.stopped_thresholds is None:
            self.stopped_thresholds = {
                'consecutive_losses': 10,  # 10 убытков подряд
                'max_drawdown': 0.30,  # Просадка > 30%
                'win_rate_drop': 0.50,  # Падение Win Rate на 50%
                'profit_factor_drop': 0.70,  # Падение Profit Factor на 70%
                'drift_score': 0.20,  # Дрифт модели > 20%
            }
        
        if self.position_size_multipliers is None:
            self.position_size_multipliers = {
                'NORMAL': 1.0,    # Полный размер
                'WARNING': 0.5,   # Снижение на 50%
                'CRITICAL': 0.25, # Снижение на 75%
                'STOPPED': 0.0,   # Остановка торговли
            }
        
        if self.drift_detection is None:
            self.drift_detection = {
                'enabled': True,
                'min_samples': 20,  # Минимум последовательностей для анализа
                'max_samples': 100,  # Максимум хранимых последовательностей
                'check_interval': 10,  # Проверять дрифт каждые N сделок
            }

# Конфигурация по умолчанию
default_monitoring_config = MonitoringConfig()

