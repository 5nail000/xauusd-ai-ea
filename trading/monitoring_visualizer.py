"""
Визуализация метрик мониторинга производительности
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
from typing import Dict, Optional
from pathlib import Path


class MonitoringVisualizer:
    """
    Создает визуализации для мониторинга производительности
    """
    
    def __init__(self, output_dir: str = 'trading/monitoring_plots'):
        """
        Args:
            output_dir: Директория для сохранения графиков
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'NORMAL': '#2ecc71',    # Зеленый
            'WARNING': '#f39c12',    # Оранжевый
            'CRITICAL': '#e74c3c',   # Красный
            'STOPPED': '#8e44ad',    # Фиолетовый
        }
    
    def plot_performance_dashboard(self, 
                                   monitor_data: Dict,
                                   backtest_stats: Dict,
                                   save_path: Optional[str] = None):
        """
        Создает дашборд с основными метриками производительности
        
        Args:
            monitor_data: Данные из PerformanceMonitor.get_metrics_history()
            backtest_stats: Статистика из бэктеста
            save_path: Путь для сохранения (опционально)
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Equity Curve
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_equity_curve(ax1, monitor_data['equity_history'], monitor_data.get('status_history', []))
        
        # 2. Win Rate
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_win_rate(ax2, monitor_data['win_rate'], backtest_stats.get('win_rate', 0) / 100)
        
        # 3. Profit Factor
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_profit_factor(ax3, monitor_data['profit_factor'], backtest_stats.get('profit_factor', 1.0))
        
        # 4. Average Confidence
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_avg_confidence(ax4, monitor_data['avg_confidence'])
        
        # 5. Drawdown
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_drawdown(ax5, monitor_data['equity_history'])
        
        # 6. Anomaly Rate
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_anomaly_rate(ax6, monitor_data.get('anomaly_rate', []))
        
        # 7. Status Timeline
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_status_timeline(ax7, monitor_data.get('status_history', []))
        
        plt.suptitle('Performance Monitoring Dashboard', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Дашборд сохранен: {save_path}")
        else:
            save_path = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Дашборд сохранен: {save_path}")
        
        plt.close()
    
    def _plot_equity_curve(self, ax, equity_history: list, status_history: list):
        """График equity curve с отметками изменения статуса"""
        if not equity_history:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Equity Curve', fontweight='bold')
            return
        
        timestamps = [e['timestamp'] for e in equity_history]
        equities = [e['equity'] for e in equity_history]
        
        ax.plot(timestamps, equities, linewidth=2, color='#3498db', label='Equity')
        ax.axhline(y=equities[0], color='gray', linestyle='--', alpha=0.5, label='Начальный баланс')
        
        # Отмечаем изменения статуса
        for status_change in status_history:
            timestamp = status_change['timestamp']
            new_status = status_change['new_status']
            # Находим ближайший индекс
            if timestamps:
                idx = min(range(len(timestamps)), key=lambda i: abs((timestamps[i] - timestamp).total_seconds()))
                if idx < len(equities):
                    color = self.colors.get(new_status, 'gray')
                    ax.axvline(x=timestamps[idx], color=color, linestyle='--', alpha=0.7, linewidth=2)
                    ax.scatter([timestamps[idx]], [equities[idx]], color=color, s=100, zorder=5)
        
        ax.set_title('Equity Curve', fontweight='bold')
        ax.set_xlabel('Время')
        ax.set_ylabel('Equity ($)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат
        if len(timestamps) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_win_rate(self, ax, win_rate_history: list, expected_wr: float):
        """График Win Rate"""
        if not win_rate_history:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Win Rate', fontweight='bold')
            return
        
        x = range(len(win_rate_history))
        ax.plot(x, win_rate_history, linewidth=2, color='#2ecc71', label='Текущий Win Rate')
        ax.axhline(y=expected_wr, color='#e74c3c', linestyle='--', linewidth=2, label=f'Ожидаемый ({expected_wr:.1%})')
        
        # Зоны предупреждения
        warning_wr = expected_wr * 0.85
        critical_wr = expected_wr * 0.70
        ax.axhspan(0, critical_wr, alpha=0.2, color='red', label='Критическая зона')
        ax.axhspan(critical_wr, warning_wr, alpha=0.2, color='orange', label='Зона предупреждения')
        
        ax.set_title('Win Rate (Rolling)', fontweight='bold')
        ax.set_xlabel('Окно')
        ax.set_ylabel('Win Rate')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_profit_factor(self, ax, pf_history: list, expected_pf: float):
        """График Profit Factor"""
        if not pf_history:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Profit Factor', fontweight='bold')
            return
        
        x = range(len(pf_history))
        ax.plot(x, pf_history, linewidth=2, color='#3498db', label='Текущий PF')
        ax.axhline(y=expected_pf, color='#e74c3c', linestyle='--', linewidth=2, label=f'Ожидаемый ({expected_pf:.2f})')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Break-even')
        
        ax.set_title('Profit Factor (Rolling)', fontweight='bold')
        ax.set_xlabel('Окно')
        ax.set_ylabel('Profit Factor')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_avg_confidence(self, ax, confidence_history: list):
        """График средней уверенности"""
        if not confidence_history:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Average Confidence', fontweight='bold')
            return
        
        x = range(len(confidence_history))
        ax.plot(x, confidence_history, linewidth=2, color='#9b59b6', label='Средняя уверенность')
        
        ax.set_title('Average Confidence (Rolling)', fontweight='bold')
        ax.set_xlabel('Окно')
        ax.set_ylabel('Confidence')
        ax.set_ylim(0, 1)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown(self, ax, equity_history: list):
        """График просадки"""
        if not equity_history:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Drawdown', fontweight='bold')
            return
        
        equities = [e['equity'] for e in equity_history]
        timestamps = [e['timestamp'] for e in equity_history]
        
        # Вычисляем просадку
        peak = equities[0]
        drawdowns = []
        for equity in equities:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            drawdowns.append(dd)
        
        ax.fill_between(timestamps, 0, drawdowns, alpha=0.3, color='red', label='Drawdown')
        ax.plot(timestamps, drawdowns, linewidth=2, color='#e74c3c')
        
        # Пороги
        ax.axhline(y=0.20, color='orange', linestyle='--', alpha=0.7, label='Предупреждение (20%)')
        ax.axhline(y=0.30, color='red', linestyle='--', alpha=0.7, label='Критично (30%)')
        
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_xlabel('Время')
        ax.set_ylabel('Drawdown (%)')
        ax.set_ylim(0, max(drawdowns) * 1.1 if drawdowns else 0.1)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        if len(timestamps) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def _plot_anomaly_rate(self, ax, anomaly_history: list):
        """График частоты аномалий"""
        if not anomaly_history:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Anomaly Rate', fontweight='bold')
            return
        
        x = range(len(anomaly_history))
        ax.plot(x, anomaly_history, linewidth=2, color='#e67e22', label='Частота аномалий')
        ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Предупреждение (15%)')
        ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.7, label='Критично (25%)')
        
        ax.set_title('Anomaly Rate (Rolling)', fontweight='bold')
        ax.set_xlabel('Окно')
        ax.set_ylabel('Anomaly Rate')
        ax.set_ylim(0, max(anomaly_history) * 1.2 if anomaly_history else 0.1)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_status_timeline(self, ax, status_history: list):
        """Временная линия изменения статуса"""
        if not status_history:
            ax.text(0.5, 0.5, 'Нет изменений статуса', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Status Timeline', fontweight='bold')
            return
        
        statuses = ['NORMAL', 'WARNING', 'CRITICAL', 'STOPPED']
        y_positions = {status: i for i, status in enumerate(statuses)}
        
        for i, change in enumerate(status_history):
            timestamp = change['timestamp']
            new_status = change['new_status']
            y = y_positions.get(new_status, 0)
            color = self.colors.get(new_status, 'gray')
            
            # Линия до следующего изменения или до конца
            next_timestamp = status_history[i+1]['timestamp'] if i+1 < len(status_history) else datetime.now()
            ax.plot([timestamp, next_timestamp], [y, y], color=color, linewidth=4, alpha=0.7)
            ax.scatter([timestamp], [y], color=color, s=100, zorder=5)
        
        ax.set_title('Status Timeline', fontweight='bold')
        ax.set_xlabel('Время')
        ax.set_ylabel('Статус')
        ax.set_yticks(range(len(statuses)))
        ax.set_yticklabels(statuses)
        ax.grid(True, alpha=0.3, axis='y')
        
        if len(status_history) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    def plot_drift_analysis(self, 
                           drift_scores: list,
                           save_path: Optional[str] = None):
        """График анализа дрифта модели"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if not drift_scores:
            ax.text(0.5, 0.5, 'Нет данных о дрифте', ha='center', va='center', transform=ax.transAxes)
        else:
            x = range(len(drift_scores))
            ax.plot(x, drift_scores, linewidth=2, color='#e74c3c', label='Drift Score')
            ax.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Предупреждение (5%)')
            ax.axhline(y=0.10, color='red', linestyle='--', alpha=0.7, label='Критично (10%)')
            ax.axhline(y=0.20, color='purple', linestyle='--', alpha=0.7, label='Остановка (20%)')
            
            ax.fill_between(x, 0, drift_scores, alpha=0.3, color='red', where=np.array(drift_scores) > 0.05)
        
        ax.set_title('Model Drift Score', fontweight='bold')
        ax.set_xlabel('Проверка')
        ax.set_ylabel('Drift Score')
        ax.set_ylim(0, max(drift_scores) * 1.2 if drift_scores else 0.1)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            save_path = self.output_dir / f"drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
        print(f"📊 График дрифта сохранен: {save_path}")

