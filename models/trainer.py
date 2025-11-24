"""
Модуль для обучения Transformer модели
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict
import os

from models.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TrainingHistory
from models.evaluator import ModelEvaluator

class ModelTrainer:
    """
    Класс для обучения Transformer модели
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 scheduler_type: str = 'cosine'):
        """
        Args:
            model: Модель для обучения
            device: Устройство (CPU/GPU)
            learning_rate: Learning rate
            weight_decay: Weight decay для регуляризации
            scheduler_type: Тип learning rate scheduler
        """
        self.model = model
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss функция (CrossEntropyLoss для классификации)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        self.lr_scheduler = LearningRateScheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            T_max=100
        )
        
        # История обучения
        self.history = TrainingHistory()
        
        print(f"Модель будет обучаться на: {self.device}")
        print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Обучает модель на одной эпохе
        
        Args:
            train_loader: DataLoader для обучающих данных
        
        Returns:
            Словарь с метриками (loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for sequences, targets in pbar:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Метрики
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Обновляем progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Валидирует модель
        
        Args:
            val_loader: DataLoader для валидационных данных
        
        Returns:
            Словарь с метриками (loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for sequences, targets in pbar:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 100,
              early_stopping_patience: int = 10,
              checkpoint_path: str = 'models/checkpoints/best_model.pth',
              save_history: bool = True):
        """
        Обучает модель
        
        Args:
            train_loader: DataLoader для обучающих данных
            val_loader: DataLoader для валидационных данных
            num_epochs: Количество эпох
            early_stopping_patience: Терпение для early stopping
            checkpoint_path: Путь для сохранения лучшей модели
            save_history: Сохранять ли историю обучения
        """
        # Callbacks
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        checkpoint = ModelCheckpoint(checkpoint_path, mode='max', save_best_only=True)
        
        print(f"\nНачало обучения на {num_epochs} эпох")
        print("=" * 60)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f"\nЭпоха {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduler
            self.lr_scheduler.step(val_metrics['loss'])
            current_lr = self.lr_scheduler.get_lr()
            
            # Обновляем историю
            self.history.update(
                train_metrics['loss'],
                train_metrics['accuracy'],
                val_metrics['loss'],
                val_metrics['accuracy'],
                current_lr
            )
            
            # Сохраняем лучшую модель
            checkpoint(self.model, val_metrics['accuracy'], epoch)
            
            # Выводим метрики
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss:   {val_metrics['loss']:.4f}, Val Acc:   {val_metrics['accuracy']:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # Early stopping
            if early_stopping(val_metrics['accuracy']):
                print(f"\nEarly stopping на эпохе {epoch + 1}")
                print(f"Лучшая валидационная точность: {early_stopping.best_score:.2f}%")
                break
            
            best_val_acc = max(best_val_acc, val_metrics['accuracy'])
        
        print("\n" + "=" * 60)
        print(f"Обучение завершено!")
        print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
        print("=" * 60)
        
        # Сохраняем историю
        if save_history:
            history_path = checkpoint_path.replace('.pth', '_history.pkl')
            self.history.save(history_path)
            print(f"История обучения сохранена: {history_path}")
        
        # Загружаем лучшие веса
        checkpoint.load_best_model(self.model)
        print(f"Загружены лучшие веса из: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загружает веса модели из checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружены веса из {checkpoint_path}")
        print(f"Эпоха: {checkpoint.get('epoch', 'unknown')}")
        print(f"Score: {checkpoint.get('score', 'unknown')}")

