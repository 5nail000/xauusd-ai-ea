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
import json
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
                 model_type: str = "encoder",
                 use_class_weights: bool = False,
                 class_weights: Optional[torch.Tensor] = None):
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
            use_class_weights: Использовать ли веса классов для несбалансированных данных
            class_weights: Tensor с весами классов [weight_0, weight_1, ...]
                          Если None и use_class_weights=True, веса должны быть вычислены заранее
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
        self.use_class_weights = use_class_weights
        self.class_weights = class_weights
        self.class_weight_method = getattr(self, 'class_weight_method', None)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        
        # Оптимизатор
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss функция (CrossEntropyLoss для классификации)
        if use_class_weights and class_weights is not None:
            # Перемещаем веса на нужное устройство
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"✓ Используются веса классов: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss()
            if use_class_weights:
                print("⚠️  use_class_weights=True, но class_weights не предоставлен. Веса не используются.")
            else:
                print("✓ Веса классов не используются")
        
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
        
        # Сохраняем параметры обучения для логирования
        self.training_params = {
            'model_type': model_type,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'scheduler_type': scheduler_type,
            'use_class_weights': use_class_weights,
            'class_weight_method': self.class_weight_method,
            'class_weights': class_weights.cpu().numpy().tolist() if class_weights is not None else None,
            'device': str(self.device),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'dropout': model_config.dropout if model_config else None,
            'd_model': model_config.d_model if model_config else None,
            'n_layers': model_config.n_layers if model_config else None,
            'n_heads': model_config.n_heads if model_config else None,
            'sequence_length': model_config.sequence_length if model_config else None,
            'num_classes': model_config.num_classes if model_config else None,
            'num_features': model_config.num_features if model_config else None
        }
        
        # Логируем параметры в TensorBoard
        params_text = "\n".join([f"{k}: {v}" for k, v in self.training_params.items() if v is not None])
        self.writer.add_text('Training_Parameters', params_text, 0)
        
        # Сохраняем параметры в файл
        self._save_training_config(log_dir)
        
        # Weights & Biases
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    name=f"{model_type}_{timestamp}",
                    config=self.training_params
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
        print(f"\nПараметры обучения:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Scheduler: {scheduler_type}")
        print(f"  Class Weights: {'Да' if use_class_weights and class_weights is not None else 'Нет'}")
        if use_class_weights and class_weights is not None:
            print(f"  Веса классов: {class_weights.cpu().numpy()}")
    
    def _save_training_config(self, log_dir: str):
        """Сохраняет конфигурацию обучения в файл"""
        config_path = os.path.join(log_dir, 'training_config.json')
        
        # Преобразуем numpy arrays в списки для JSON
        config_dict = {}
        for key, value in self.training_params.items():
            if isinstance(value, np.ndarray):
                config_dict[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                config_dict[key] = float(value)
            else:
                config_dict[key] = value
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        print(f"  Конфигурация обучения сохранена: {config_path}")
    
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
        # Создаем директорию для анализа обучения
        analysis_dir = 'workspace/analysis-of-train'
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Определяем базовое имя модели из checkpoint_path
        model_name = os.path.basename(checkpoint_path).replace('.pth', '')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_subdir = os.path.join(analysis_dir, f"{model_name}_{timestamp}")
        os.makedirs(analysis_subdir, exist_ok=True)
        
        # Пути для сохранения
        history_csv_path = os.path.join(analysis_subdir, f'{model_name}_history.csv')
        history_pkl_path = os.path.join(analysis_subdir, f'{model_name}_history.pkl')
        plot_path = os.path.join(analysis_subdir, f'{model_name}_training_curves.png')
        text_log_path = os.path.join(analysis_subdir, f'{model_name}_training_log.txt')
        
        # Открываем файл для текстового лога
        log_file = open(text_log_path, 'w', encoding='utf-8')
        
        def log_print(*args, **kwargs):
            """Выводит в консоль и в файл"""
            print(*args, **kwargs)
            print(*args, file=log_file, **kwargs)
            log_file.flush()  # Гарантируем запись в файл
        
        # Callbacks
        early_stopping = EarlyStopping(patience=early_stopping_patience, mode='max')
        checkpoint = ModelCheckpoint(checkpoint_path, mode='max', save_best_only=True, model_config=self.model_config)
        
        log_print(f"\nНачало обучения на {num_epochs} эпох")
        log_print(f"Анализ обучения сохраняется в: {analysis_subdir}")
        log_print("=" * 60)
        
        best_val_acc = 0.0
        interrupted = False
        
        try:
            for epoch in range(num_epochs):
                log_print(f"\nЭпоха {epoch + 1}/{num_epochs}")
                log_print("-" * 60)
                
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
                log_print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
                log_print(f"Val Loss:   {val_metrics['loss']:.4f}, Val Acc:   {val_metrics['accuracy']:.2f}%")
                log_print(f"Learning Rate: {current_lr:.6f}")
                
                # Сохраняем историю после каждой эпохи (для защиты от прерываний)
                if save_history and len(self.history.history['train_loss']) > 0:
                    try:
                        self.history.save_to_csv(history_csv_path)
                        self.history.save(history_pkl_path)
                    except Exception as e:
                        log_print(f"⚠️  Ошибка при сохранении истории: {e}")
                
                # Early stopping
                if early_stopping(val_metrics['accuracy']):
                    log_print(f"\nEarly stopping на эпохе {epoch + 1}")
                    log_print(f"Лучшая валидационная точность: {early_stopping.best_score:.2f}%")
                    break
                
                best_val_acc = max(best_val_acc, val_metrics['accuracy'])
        
        except KeyboardInterrupt:
            interrupted = True
            log_print("\n" + "=" * 60)
            log_print("⚠️  ОБУЧЕНИЕ ПРЕРВАНО ПОЛЬЗОВАТЕЛЕМ (Ctrl+C)")
            log_print("=" * 60)
            log_print(f"Обработано эпох: {len(self.history.history['train_loss'])}")
            log_print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
        
        except Exception as e:
            interrupted = True
            log_print("\n" + "=" * 60)
            log_print(f"❌ ОШИБКА ПРИ ОБУЧЕНИИ: {e}")
            log_print("=" * 60)
            import traceback
            log_print(traceback.format_exc())
            log_print(f"Обработано эпох: {len(self.history.history['train_loss'])}")
            log_print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
        
        finally:
            # Гарантированно сохраняем все данные
            if not interrupted:
                log_print("\n" + "=" * 60)
                log_print(f"Обучение завершено!")
                log_print(f"Лучшая валидационная точность: {best_val_acc:.2f}%")
                log_print("=" * 60)
            
            # Сохраняем историю (если есть данные)
            if save_history and len(self.history.history['train_loss']) > 0:
                try:
                    log_print(f"\nСохранение истории обучения...")
                    self.history.save(history_pkl_path)
                    log_print(f"  ✓ История сохранена: {history_pkl_path}")
                    
                    self.history.save_to_csv(history_csv_path)
                    log_print(f"  ✓ CSV сохранен: {history_csv_path}")
                    
                    # Строим графики
                    try:
                        self.history.plot_history(save_path=plot_path, show=False)
                        log_print(f"  ✓ Графики сохранены: {plot_path}")
                    except Exception as e:
                        log_print(f"  ⚠️  Ошибка при создании графиков: {e}")
                    
                    # Анализ переобученности
                    try:
                        overfitting_analysis = self.history.analyze_overfitting()
                        if overfitting_analysis:
                            log_print("\n" + "=" * 60)
                            log_print("АНАЛИЗ ПЕРЕОБУЧЕННОСТИ")
                            log_print("=" * 60)
                            log_print(f"Переобученность: {overfitting_analysis['overfitting_severity']}")
                            log_print(f"Финальный gap по accuracy: {overfitting_analysis['final_acc_gap']:.2f}%")
                            log_print(f"  Train Acc: {overfitting_analysis['final_train_acc']:.2f}%")
                            log_print(f"  Val Acc:   {overfitting_analysis['final_val_acc']:.2f}%")
                            log_print(f"Финальный gap по loss: {overfitting_analysis['final_loss_gap']:.4f}")
                            log_print(f"  Train Loss: {overfitting_analysis['final_train_loss']:.4f}")
                            log_print(f"  Val Loss:   {overfitting_analysis['final_val_loss']:.4f}")
                            log_print(f"\nЛучшая валидационная точность: {overfitting_analysis['best_val_acc']:.2f}%")
                            log_print(f"Достигнута на эпохе: {overfitting_analysis['best_val_acc_epoch']}")
                            
                            if overfitting_analysis['is_overfitting']:
                                log_print(f"\n⚠️  Обнаружена переобученность!")
                                log_print(f"   Рекомендации:")
                                log_print(f"   - Увеличить dropout")
                                log_print(f"   - Добавить регуляризацию")
                                log_print(f"   - Увеличить размер обучающей выборки")
                                log_print(f"   - Уменьшить сложность модели")
                            else:
                                log_print(f"\n✓ Переобученность не обнаружена")
                            log_print("=" * 60)
                    except Exception as e:
                        log_print(f"  ⚠️  Ошибка при анализе переобученности: {e}")
                except Exception as e:
                    log_print(f"❌ Критическая ошибка при сохранении истории: {e}")
                    import traceback
                    log_print(traceback.format_exc())
            
            # Загружаем лучшие веса
            try:
                checkpoint.load_best_model(self.model)
                log_print(f"Загружены лучшие веса из: {checkpoint_path}")
            except Exception as e:
                log_print(f"⚠️  Ошибка при загрузке лучших весов: {e}")
            
            # Закрываем TensorBoard writer
            try:
                self.writer.flush()  # Гарантируем запись всех данных
                self.writer.close()
                log_print(f"TensorBoard логи сохранены")
            except Exception as e:
                log_print(f"⚠️  Ошибка при закрытии TensorBoard writer: {e}")
            
            # Завершаем W&B сессию
            if self.use_wandb and self.wandb:
                try:
                    self.wandb.finish()
                    log_print(f"W&B сессия завершена")
                except Exception as e:
                    log_print(f"⚠️  Ошибка при завершении W&B: {e}")
            
            # Финальное сообщение
            if interrupted:
                log_print(f"\n⚠️  Обучение было прервано, но все доступные данные сохранены в: {analysis_subdir}")
            else:
                log_print(f"\n✓ Все данные сохранены в: {analysis_subdir}")
            
            # Закрываем файл лога в самом конце
            log_file.close()
            
            # Выводим финальное сообщение в консоль
            if interrupted:
                print(f"\n⚠️  Обучение было прервано, но все доступные данные сохранены в: {analysis_subdir}")
            else:
                print(f"\n✓ Все данные сохранены в: {analysis_subdir}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Загружает веса модели из checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Загружены веса из {checkpoint_path}")
        print(f"Эпоха: {checkpoint.get('epoch', 'unknown')}")
        print(f"Score: {checkpoint.get('score', 'unknown')}")

