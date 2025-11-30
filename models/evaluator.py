"""
Модуль для оценки модели
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Класс для оценки обученной модели
    """
    
    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Args:
            model: Обученная модель
            device: Устройство (CPU/GPU)
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Оценивает модель на данных
        
        Args:
            data_loader: DataLoader с данными
        
        Returns:
            Словарь с метриками
        """
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for sequences, targets in data_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Вычисляем метрики
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Метрики по классам
        precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'predictions': all_predictions,
            'targets': all_targets
        }
        
        return metrics
    
    def get_confusion_matrix(self, data_loader: DataLoader, 
                            class_names: list = None) -> np.ndarray:
        """
        Возвращает confusion matrix
        
        Args:
            data_loader: DataLoader с данными
            class_names: Названия классов
        
        Returns:
            Confusion matrix
        """
        metrics = self.evaluate(data_loader)
        cm = confusion_matrix(metrics['targets'], metrics['predictions'])
        return cm
    
    def print_classification_report(self, data_loader: DataLoader,
                                   class_names: list = None):
        """
        Выводит classification report
        
        Args:
            data_loader: DataLoader с данными
            class_names: Названия классов
        """
        metrics = self.evaluate(data_loader)
        
        if class_names is None:
            # Автоматически определяем количество классов из данных
            unique_classes = sorted(set(metrics['targets']) | set(metrics['predictions']))
            class_name_map = {
                0: 'Uncertainty',
                1: 'Breakout Up',
                2: 'Breakout Down',
                3: 'Bounce Up',
                4: 'Bounce Down'
            }
            class_names = [class_name_map.get(i, f'Class {i}') for i in unique_classes]
        
        report = classification_report(
            metrics['targets'],
            metrics['predictions'],
            target_names=class_names,
            zero_division=0
        )
        
        print("\nClassification Report:")
        print("=" * 60)
        print(report)
        print("=" * 60)
    
    def plot_confusion_matrix(self, data_loader: DataLoader,
                             class_names: list = None,
                             save_path: str = None,
                             show: bool = True):
        """
        Визуализирует confusion matrix
        
        Args:
            data_loader: DataLoader с данными
            class_names: Названия классов
            save_path: Путь для сохранения графика
            show: Показывать ли график (по умолчанию: True)
        """
        cm = self.get_confusion_matrix(data_loader, class_names)
        
        if class_names is None:
            # Автоматически определяем количество классов из данных
            metrics = self.evaluate(data_loader)
            unique_classes = sorted(set(metrics['targets']) | set(metrics['predictions']))
            class_name_map = {
                0: 'Uncertainty',
                1: 'Breakout Up',
                2: 'Breakout Down',
                3: 'Bounce Up',
                4: 'Bounce Down'
            }
            class_names = [class_name_map.get(i, f'Class {i}') for i in unique_classes]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix сохранена: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def get_class_distribution(self, data_loader: DataLoader) -> pd.Series:
        """
        Возвращает распределение классов в предсказаниях и реальных метках
        
        Args:
            data_loader: DataLoader с данными
        
        Returns:
            DataFrame с распределением
        """
        metrics = self.evaluate(data_loader)
        
        true_dist = pd.Series(metrics['targets']).value_counts().sort_index()
        pred_dist = pd.Series(metrics['predictions']).value_counts().sort_index()
        
        df = pd.DataFrame({
            'True': true_dist,
            'Predicted': pred_dist
        })
        
        # Автоматически определяем названия классов
        unique_classes = sorted(set(metrics['targets']) | set(metrics['predictions']))
        class_name_map = {
            0: 'Uncertainty',
            1: 'Breakout Up',
            2: 'Breakout Down',
            3: 'Bounce Up',
            4: 'Bounce Down'
        }
        class_names = [class_name_map.get(i, f'Class {i}') for i in unique_classes]
        df.index = class_names
        
        return df
    
    def evaluate_all_splits(self, train_loader: DataLoader,
                           val_loader: DataLoader,
                           test_loader: DataLoader) -> Dict[str, Dict]:
        """
        Оценивает модель на всех выборках
        
        Args:
            train_loader: DataLoader для train
            val_loader: DataLoader для validation
            test_loader: DataLoader для test
        
        Returns:
            Словарь с метриками для каждой выборки
        """
        results = {}
        
        print("\n" + "=" * 60)
        print("Оценка модели на всех выборках")
        print("=" * 60)
        
        for name, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
            print(f"\n{name}:")
            print("-" * 60)
            metrics = self.evaluate(loader)
            
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision_weighted']:.4f}")
            print(f"Recall:    {metrics['recall_weighted']:.4f}")
            print(f"F1-Score:  {metrics['f1_weighted']:.4f}")
            
            print("\nПо классам:")
            # Автоматически определяем количество классов из данных
            unique_classes = sorted(set(metrics['targets']) | set(metrics['predictions']))
            class_name_map = {
                0: 'Uncertainty',
                1: 'Breakout Up',
                2: 'Breakout Down',
                3: 'Bounce Up',
                4: 'Bounce Down'
            }
            class_names = [class_name_map.get(i, f'Class {i}') for i in unique_classes]
            
            for i, class_name in enumerate(class_names):
                if i < len(metrics['precision_per_class']):
                    print(f"  {class_name}:")
                    print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
                    print(f"    Recall:    {metrics['recall_per_class'][i]:.4f}")
                    print(f"    F1:        {metrics['f1_per_class'][i]:.4f}")
            
            results[name.lower()] = metrics
        
        return results

