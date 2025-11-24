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
    
    def __init__(self, filepath: str, mode: str = 'min', save_best_only: bool = True):
        """
        Args:
            filepath: Путь для сохранения модели
            mode: 'min' для минимизации метрики, 'max' для максимизации
            save_best_only: Сохранять только лучшие веса
        """
        self.filepath = filepath
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.best_epoch = 0
    
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
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'score': self.best_score
        }, filepath)
    
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

