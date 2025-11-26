"""
Модуль для обучения Transformer модели
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict
import os
from datetime import datetime

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
                 scheduler_type: str = 'cosine',
                 model_config=None,
                 use_wandb: bool = False,
                 wandb_project: str = "xauusd-ai-ea",
                 model_type: str = "encoder"):
        """
        Args:
            model: Модель для обучения
            device: Устройство (CPU/GPU)
            learning_rate: Learning rate
            weight_decay: Weight decay для регуляризации
            scheduler_type: Тип learning rate scheduler
            model_config: Конфигурация модели для сохранения в checkpoint
            use_wandb: Использовать ли Weights & Biases для логирования
            wandb_project: Название проекта в W&B
            model_type: Тип модели (для логирования)
        """
        self.model = model
        self.device = device if device is not None else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)
        self.model_config = model_config
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.model_type = model_type
        
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
        
        # TensorBoard writer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f'workspace/models/logs/{model_type}_{timestamp}'
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # Weights & Biases
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=f"{model_type}_{timestamp}",
                    config={
                        'model_type': model_type,
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'scheduler_type': scheduler_type,
                        'device': str(self.device)
                    }
                )
                self.wandb = wandb
            except ImportError:
                print("⚠️  W&B не установлен. Установите: pip install wandb")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None
        
        print(f"Модель будет обучаться на: {self.device}")
        print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
        if use_wandb:
            print(f"W&B логирование: включено (проект: {wandb_project})")
        print(f"TensorBoard логи: {log_dir}")
    
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
        global_step = 0
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
            accuracy = 100 * correct / total
            
            # Логируем метрики по батчам в TensorBoard
            self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
            self.writer.add_scalar('Accuracy/Train_Batch', accuracy, global_step)
            
            # Обновляем progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.2f}%'
            })
            
            global_step += 1
        
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
              checkpoint_path: str = 'workspace/models/checkpoints/best_model.pth',
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
        checkpoint = ModelCheckpoint(checkpoint_path, mode='max', save_best_only=True, model_config=self.model_config)
        
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
            
            # Логируем метрики по эпохам в TensorBoard
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Val', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Логируем в W&B
            if self.use_wandb and self.wandb:
                self.wandb.log({
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_acc': val_metrics['accuracy'],
                    'lr': current_lr
                }, step=epoch)
            
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
            # Определяем базовое имя модели из checkpoint_path
            import os
            model_name = os.path.basename(checkpoint_path).replace('.pth', '')
            
            # Сохраняем историю в workspace/models/metrics/
            os.makedirs('workspace/models/metrics', exist_ok=True)
            
            history_path = f'workspace/models/metrics/{model_name}_history.pkl'
            self.history.save(history_path)
            print(f"История обучения сохранена: {history_path}")
            
            # Сохраняем в CSV
            csv_path = f'workspace/models/metrics/{model_name}_history.csv'
            self.history.save_to_csv(csv_path)
            
            # Строим графики
            plot_path = f'workspace/models/metrics/{model_name}_training_curves.png'
            self.history.plot_history(save_path=plot_path, show=False)
            
            # Анализ переобученности
            overfitting_analysis = self.history.analyze_overfitting()
            if overfitting_analysis:
                print("\n" + "=" * 60)
                print("АНАЛИЗ ПЕРЕОБУЧЕННОСТИ")
                print("=" * 60)
                print(f"Переобученность: {overfitting_analysis['overfitting_severity']}")
                print(f"Финальный gap по accuracy: {overfitting_analysis['final_acc_gap']:.2f}%")
                print(f"  Train Acc: {overfitting_analysis['final_train_acc']:.2f}%")
                print(f"  Val Acc:   {overfitting_analysis['final_val_acc']:.2f}%")
                print(f"Финальный gap по loss: {overfitting_analysis['final_loss_gap']:.4f}")
                print(f"  Train Loss: {overfitting_analysis['final_train_loss']:.4f}")
                print(f"  Val Loss:   {overfitting_analysis['final_val_loss']:.4f}")
                print(f"\nЛучшая валидационная точность: {overfitting_analysis['best_val_acc']:.2f}%")
                print(f"Достигнута на эпохе: {overfitting_analysis['best_val_acc_epoch']}")
                
                if overfitting_analysis['is_overfitting']:
                    print(f"\n⚠️  Обнаружена переобученность!")
                    print(f"   Рекомендации:")
                    print(f"   - Увеличить dropout")
                    print(f"   - Добавить регуляризацию")
                    print(f"   - Увеличить размер обучающей выборки")
                    print(f"   - Уменьшить сложность модели")
                else:
                    print(f"\n✓ Переобученность не обнаружена")
                print("=" * 60)
        
        # Загружаем лучшие веса
        checkpoint.load_best_model(self.model)
        print(f"Загружены лучшие веса из: {checkpoint_path}")
        
        # Закрываем TensorBoard writer
        self.writer.close()
        
        # Завершаем W&B сессию
        if self.use_wandb and self.wandb:
            self.wandb.finish()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загружает веса модели из checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружены веса из {checkpoint_path}")
        print(f"Эпоха: {checkpoint.get('epoch', 'unknown')}")
        print(f"Score: {checkpoint.get('score', 'unknown')}")

