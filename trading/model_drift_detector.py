"""
Обнаружение дрифта модели (изменение распределения данных)
"""
import numpy as np
from typing import Dict, Optional
from config.monitoring_config import MonitoringConfig


class ModelDriftDetector:
    """
    Обнаружение дрифта модели (изменение распределения данных)
    
    Дрифт модели происходит, когда распределение входных данных
    или поведение рынка меняется так, что модель, обученная на старых данных,
    перестает работать на новых.
    
    Примеры:
    - Изменилась волатильность рынка
    - Изменился тренд (бычий -> медвежий)
    - Новые рыночные условия (кризис, новости)
    """
    
    def __init__(self,
                 training_feature_stats: Dict,
                 config: Optional[MonitoringConfig] = None):
        """
        Args:
            training_feature_stats: Статистика фичей из обучающей выборки
                Формат: {feature_name: {'mean': float, 'std': float}}
            config: Конфигурация мониторинга
        """
        self.training_stats = training_feature_stats
        self.config = config if config else MonitoringConfig()
        
        # История последовательностей для анализа
        self.recent_sequences = []
        self.drift_scores = []
        self.feature_names = list(training_feature_stats.keys()) if training_feature_stats else []
        
        # Текущая оценка дрифта
        self.current_drift_score = 0.0
        
    def add_sequence(self, sequence: np.ndarray, feature_names: Optional[list] = None):
        """
        Добавляет новую последовательность для анализа дрифта
        
        Args:
            sequence: Массив последовательности [seq_len, n_features]
            feature_names: Названия фичей (опционально)
        """
        if feature_names:
            self.feature_names = feature_names
        
        # Берем последний временной шаг последовательности
        last_step = sequence[-1] if len(sequence.shape) > 1 else sequence
        
        self.recent_sequences.append(last_step)
        
        # Храним только последние N последовательностей
        max_samples = self.config.drift_detection['max_samples']
        if len(self.recent_sequences) > max_samples:
            self.recent_sequences.pop(0)
        
        # Вычисляем дрифт если накопилось достаточно данных
        min_samples = self.config.drift_detection['min_samples']
        if len(self.recent_sequences) >= min_samples:
            self.current_drift_score = self._calculate_drift_score()
            self.drift_scores.append(self.current_drift_score)
    
    def _calculate_drift_score(self) -> float:
        """
        Вычисляет оценку дрифта (0-1, где 1 = максимальный дрифт)
        
        Использует сравнение распределений фичей:
        - Среднее значение (mean)
        - Стандартное отклонение (std)
        """
        if not self.recent_sequences or not self.training_stats:
            return 0.0
        
        # Собираем значения по каждой фиче
        feature_values = {}
        for i, feature_name in enumerate(self.feature_names):
            if i >= len(self.recent_sequences[0]):
                continue
            
            values = [seq[i] for seq in self.recent_sequences if i < len(seq)]
            if values:
                feature_values[feature_name] = np.array(values)
        
        # Вычисляем дрифт для каждой фичи
        drift_scores = []
        
        for feature_name, recent_values in feature_values.items():
            if feature_name not in self.training_stats:
                continue
            
            training_mean = self.training_stats[feature_name]['mean']
            training_std = self.training_stats[feature_name]['std']
            
            # Избегаем деления на ноль
            if training_std < 1e-8:
                continue
            
            # Вычисляем статистики для текущих данных
            recent_mean = np.mean(recent_values)
            recent_std = np.std(recent_values)
            
            # Нормализованное расстояние между средними
            mean_diff = abs(recent_mean - training_mean) / training_std
            
            # Нормализованное расстояние между стандартными отклонениями
            std_diff = abs(recent_std - training_std) / training_std if training_std > 0 else 0
            
            # Комбинируем метрики (можно использовать разные веса)
            feature_drift = (mean_diff * 0.6 + std_diff * 0.4)
            
            # Нормализуем до 0-1 (используем сигмоиду для сглаживания)
            feature_drift_normalized = 1 / (1 + np.exp(-feature_drift * 2))
            
            drift_scores.append(feature_drift_normalized)
        
        # Средний дрифт по всем фичам
        avg_drift = np.mean(drift_scores) if drift_scores else 0.0
        
        return min(avg_drift, 1.0)  # Ограничиваем до 1.0
    
    def has_drift(self, threshold: Optional[float] = None) -> bool:
        """
        Проверяет, есть ли значительный дрифт
        
        Args:
            threshold: Порог для обнаружения дрифта (по умолчанию из конфига)
        """
        if threshold is None:
            threshold = self.config.warning_thresholds.get('drift_score', 0.05)
        
        return self.current_drift_score > threshold
    
    def get_drift_report(self) -> Dict:
        """Возвращает отчет о дрифте"""
        return {
            'drift_score': self.current_drift_score,
            'has_drift': self.has_drift(),
            'samples_analyzed': len(self.recent_sequences),
            'min_samples_required': self.config.drift_detection['min_samples'],
            'drift_history': self.drift_scores[-50:] if len(self.drift_scores) > 50 else self.drift_scores,
        }

