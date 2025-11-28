"""
Callbacks для обучения модели
"""
import torch
import os
from typing import Optional, Dict
import numpy as np

class EarlyStopping:
    """Early stopping для остановки обучения при отсутствии улучшений"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Количество эпох без улучшения перед остановкой
            min_delta: Минимальное изменение для считания улучшением
            mode: 'min' для минимизации метрики, 'max' для максимизации
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Проверяет, нужно ли остановить обучение
        
        Args:
            score: Текущее значение метрики
        
        Returns:
            True если нужно остановить обучение
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Проверяет, лучше ли текущее значение"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def reset(self):
        """Сбрасывает состояние"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

class ModelCheckpoint:
    """Callback для сохранения лучших весов модели"""
    
    def __init__(self, filepath: str, mode: str = 'min', save_best_only: bool = True, 
                 model_config=None, training_params: Optional[Dict] = None):
        """
        Args:
            filepath: Путь для сохранения модели
            mode: 'min' для минимизации метрики, 'max' для максимизации
            save_best_only: Сохранять только лучшие веса
            model_config: Конфигурация модели для сохранения
            training_params: Параметры обучения для сохранения в checkpoint
        """
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.best_epoch = 0
        self.model_config = model_config
        self.training_params = training_params
    
    def __call__(self, model: torch.nn.Module, score: float, epoch: int):
        """
        Сохраняет модель если метрика улучшилась
        
        Args:
            model: Модель для сохранения
            score: Текущее значение метрики
            epoch: Номер эпохи
        """
        if self.best_score is None:
            self.best_score = score
            self._save_model(model, epoch)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            if self.save_best_only:
                self._save_model(model, epoch)
        elif not self.save_best_only:
            # Сохраняем каждую эпоху
            self._save_model(model, epoch, suffix=f'_epoch_{epoch}')
    
    def _is_better(self, current: float, best: float) -> bool:
        """Проверяет, лучше ли текущее значение"""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _save_model(self, model: torch.nn.Module, epoch: int, suffix: str = ''):
        """Сохраняет модель"""
        filepath = self.filepath.replace('.pth', f'{suffix}.pth')
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'score': self.best_score
        }
        
        # Сохраняем параметры обучения
        if self.training_params is not None:
            # Преобразуем numpy arrays в списки для сохранения
            training_params_save = {}
            for key, value in self.training_params.items():
                if isinstance(value, np.ndarray):
                    training_params_save[key] = value.tolist()
                elif isinstance(value, (np.integer, np.floating)):
                    training_params_save[key] = float(value)
                else:
                    training_params_save[key] = value
            checkpoint['training_params'] = training_params_save
        
        # Сохраняем конфигурацию модели, если она предоставлена
        if self.model_config is not None:
            # Преобразуем dataclass в словарь для сохранения
            if hasattr(self.model_config, '__dataclass_fields__'):
                # Для dataclass - используем asdict если доступен, иначе вручную
                try:
                    from dataclasses import asdict
                    checkpoint['model_config'] = asdict(self.model_config)
                except (ImportError, TypeError):
                    # Fallback: вручную собираем словарь
                    checkpoint['model_config'] = {
                        field.name: getattr(self.model_config, field.name)
                        for field in self.model_config.__dataclass_fields__.values()
                    }
            elif hasattr(self.model_config, '__dict__'):
                # Для обычных объектов
                checkpoint['model_config'] = self.model_config.__dict__
        
        torch.save(checkpoint, filepath)
    
    def load_best_model(self, model: torch.nn.Module) -> Dict:
        """Загружает лучшие веса модели"""
        checkpoint = torch.load(self.filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

class LearningRateScheduler:
    """Callback для изменения learning rate"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, 
                 scheduler_type: str = 'cosine',
                 T_max: int = 100,
                 eta_min: float = 1e-6):
        """
        Args:
            optimizer: Оптимизатор
            scheduler_type: Тип scheduler ('cosine', 'step', 'plateau')
            T_max: Максимальное количество эпох для cosine
            eta_min: Минимальный learning rate
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        elif scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        else:
            raise ValueError(f"Неизвестный тип scheduler: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """Делает шаг scheduler"""
        if self.scheduler_type == 'plateau':
            if metric is None:
                raise ValueError("Для plateau scheduler нужна метрика")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Возвращает текущий learning rate"""
        return self.optimizer.param_groups[0]['lr']

class TrainingHistory:
    """Хранит историю обучения"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def update(self, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float, lr: float):
        """Обновляет историю"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
    
    def get_best_epoch(self, metric: str = 'val_acc', mode: str = 'max') -> int:
        """Возвращает номер эпохи с лучшей метрикой"""
        values = self.history[metric]
        if mode == 'max':
            return np.argmax(values)
        else:
            return np.argmin(values)
    
    def save(self, filepath: str):
        """Сохраняет историю"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)
    
    def load(self, filepath: str):
        """Загружает историю"""
        import pickle
        with open(filepath, 'rb') as f:
            self.history = pickle.load(f)
    
    def save_to_csv(self, filepath: str):
        """Сохраняет историю в CSV"""
        import pandas as pd
        
        # Создаем DataFrame
        df = pd.DataFrame({
            'epoch': range(1, len(self.history['train_loss']) + 1),
            'train_loss': self.history['train_loss'],
            'train_acc': self.history['train_acc'],
            'val_loss': self.history['val_loss'],
            'val_acc': self.history['val_acc'],
            'lr': self.history['lr']
        })
        
        # Добавляем метрики переобученности
        df['loss_gap'] = df['train_loss'] - df['val_loss']
        df['acc_gap'] = df['val_acc'] - df['train_acc']
        df['overfitting_score'] = df['acc_gap']  # Отрицательное значение = переобучение
        
        df.to_csv(filepath, index=False)
        print(f"История обучения сохранена в CSV: {filepath}")
    
    def analyze_overfitting(self) -> dict:
        """Анализирует переобученность модели"""
        if not self.history['train_loss']:
            return {}
        
        import numpy as np
        
        # Берем последние значения
        final_train_loss = self.history['train_loss'][-1]
        final_val_loss = self.history['val_loss'][-1]
        final_train_acc = self.history['train_acc'][-1]
        final_val_acc = self.history['val_acc'][-1]
        
        # Вычисляем gap
        loss_gap = final_train_loss - final_val_loss
        acc_gap = final_val_acc - final_train_acc
        
        # Средние значения
        avg_train_loss = np.mean(self.history['train_loss'])
        avg_val_loss = np.mean(self.history['val_loss'])
        avg_train_acc = np.mean(self.history['train_acc'])
        avg_val_acc = np.mean(self.history['val_acc'])
        
        # Определяем переобученность
        is_overfitting = False
        overfitting_severity = "Нет"
        
        if acc_gap < -5:  # Train accuracy значительно выше val
            is_overfitting = True
            if acc_gap < -20:
                overfitting_severity = "Сильное"
            elif acc_gap < -10:
                overfitting_severity = "Умеренное"
            else:
                overfitting_severity = "Слабое"
        
        analysis = {
            'is_overfitting': is_overfitting,
            'overfitting_severity': overfitting_severity,
            'final_loss_gap': loss_gap,
            'final_acc_gap': acc_gap,
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
            'avg_train_acc': avg_train_acc,
            'avg_val_acc': avg_val_acc,
            'best_val_acc': max(self.history['val_acc']) if self.history['val_acc'] else 0,
            'best_val_acc_epoch': self.get_best_epoch('val_acc', 'max') + 1
        }
        
        return analysis
    
    def plot_history(self, save_path: str = None, show: bool = True):
        """Строит графики обучения"""
        import matplotlib.pyplot as plt
        
        if not self.history['train_loss']:
            print("История пуста, нечего строить")
            return
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Создаем фигуру с 3 подграфиками
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Accuracy
        axes[1].plot(epochs, self.history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. Learning Rate
        axes[2].plot(epochs, self.history['lr'], 'g-', label='Learning Rate', linewidth=2)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики обучения сохранены: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

