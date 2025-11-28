"""
Пример использования Walk-Forward Validation

Этот скрипт демонстрирует, как использовать Walk-Forward Validation
для более реалистичной оценки модели на временных рядах.
"""
import pandas as pd
import torch
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.walk_forward import WalkForwardValidator
from models.model_factory import create_model, get_model_config
from models.data_loader import create_dataloaders, compute_class_weights
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator


def train_model_for_fold(train_df: pd.DataFrame, 
                         val_df: pd.DataFrame,
                         model_type: str = 'encoder',
                         batch_size: int = 32,
                         num_epochs: int = 50,
                         use_class_weights: bool = False,
                         class_weight_method: str = 'balanced',
                         **kwargs) -> torch.nn.Module:
    """
    Функция обучения модели для одного окна walk-forward validation
    
    Args:
        train_df: Обучающие данные для этого окна
        val_df: Валидационные данные для этого окна
        model_type: Тип модели ('encoder' или 'timeseries')
        batch_size: Размер батча
        num_epochs: Количество эпох
        use_class_weights: Использовать ли веса классов
        class_weight_method: Метод вычисления весов
        **kwargs: Дополнительные параметры
    
    Returns:
        Обученная модель
    """
    # Создаем DataLoader'ы
    train_loader, val_loader, _, seq_gen = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=val_df,  # Для этого примера используем val как test
        sequence_length=60,
        batch_size=batch_size,
        target_column='signal_class'
    )
    
    num_features = train_loader.dataset.sequences.shape[2]
    
    # Вычисляем веса классов (если нужно)
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(
            train_df,
            target_column='signal_class',
            method=class_weight_method
        )
    
    # Создаем модель
    if model_type == 'encoder':
        config = get_model_config(
            model_type='encoder',
            num_features=num_features,
            num_classes=5,
            sequence_length=60,
            d_model=256,
            n_layers=4,
            n_heads=8,
            dropout=0.1
        )
    else:
        config = get_model_config(
            model_type='timeseries',
            num_features=num_features,
            num_classes=5,
            sequence_length=60,
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            use_temporal_encoding=True
        )
    
    model = create_model(config)
    
    # Создаем trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=1e-4,
        weight_decay=1e-5,
        scheduler_type='cosine',
        model_config=config,
        model_type=model_type,
        use_class_weights=use_class_weights,
        class_weights=class_weights
    )
    
    # Обучаем модель
    checkpoint_path = f'workspace/models/checkpoints/walkforward_{model_type}_fold_temp.pth'
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=5,  # Меньше patience для быстрого примера
        checkpoint_path=checkpoint_path,
        save_history=False  # Не сохраняем историю для каждого окна
    )
    
    # Загружаем лучшую модель
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Удаляем временный checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return model


def evaluate_model_for_fold(model: torch.nn.Module,
                            test_df: pd.DataFrame,
                            model_type: str = 'encoder',
                            batch_size: int = 32,
                            **kwargs) -> dict:
    """
    Функция оценки модели на тестовом наборе для одного окна
    
    Args:
        model: Обученная модель
        test_df: Тестовые данные для этого окна
        model_type: Тип модели
        batch_size: Размер батча
        **kwargs: Дополнительные параметры
    
    Returns:
        Словарь с метриками
    """
    # Создаем DataLoader для теста
    _, _, test_loader, _ = create_dataloaders(
        train_df=test_df.iloc[:100],  # Заглушка для train
        val_df=test_df.iloc[:100],    # Заглушка для val
        test_df=test_df,
        sequence_length=60,
        batch_size=batch_size,
        target_column='signal_class'
    )
    
    # Оцениваем модель
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(test_loader)
    
    return metrics


def main():
    """
    Основная функция для демонстрации walk-forward validation
    """
    print("=" * 80)
    print("ПРИМЕР: WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    # Параметры
    model_type = 'encoder'  # или 'timeseries'
    batch_size = 32
    num_epochs = 30  # Меньше эпох для быстрого примера
    use_class_weights = True
    class_weight_method = 'balanced'
    
    # Параметры walk-forward validation
    train_size = 60   # 60 дней обучения (примерно 2 месяца)
    val_size = 10     # 10 дней валидации
    test_size = 10    # 10 дней тестирования
    step_size = 5     # Новое окно каждые 5 дней
    
    # Загружаем данные
    print("\n1. Загрузка данных...")
    try:
        df = pd.read_csv('workspace/prepared/features/gold_train.csv', 
                         index_col=0, parse_dates=True)
        print(f"   Загружено {len(df)} образцов")
        print(f"   Период: {df.index[0]} - {df.index[-1]}")
    except FileNotFoundError:
        print("❌ Ошибка: Файл данных не найден!")
        print("   Сначала запустите: python prepare_gold_data.py --months 3")
        return
    
    # Проверяем, достаточно ли данных
    min_required = train_size + val_size + test_size
    if len(df) < min_required:
        print(f"❌ Ошибка: Недостаточно данных!")
        print(f"   Требуется минимум {min_required} образцов, доступно {len(df)}")
        return
    
    # Создаем валидатор
    print("\n2. Создание Walk-Forward Validator...")
    validator = WalkForwardValidator(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        step_size=step_size,
        use_days=False  # Используем образцы, не дни
    )
    
    # Выполняем walk-forward validation
    print("\n3. Выполнение Walk-Forward Validation...")
    print("   Это может занять некоторое время...\n")
    
    try:
        results = validator.validate(
            df=df,
            train_function=lambda train_df, val_df, **kw: train_model_for_fold(
                train_df, val_df,
                model_type=model_type,
                batch_size=batch_size,
                num_epochs=num_epochs,
                use_class_weights=use_class_weights,
                class_weight_method=class_weight_method,
                **kw
            ),
            evaluate_function=lambda model, test_df, **kw: evaluate_model_for_fold(
                model, test_df,
                model_type=model_type,
                batch_size=batch_size,
                **kw
            ),
            model_type=model_type,
            batch_size=batch_size,
            num_epochs=num_epochs,
            use_class_weights=use_class_weights,
            class_weight_method=class_weight_method
        )
        
        # Выводим результаты
        print("\n" + "=" * 80)
        print("РЕЗУЛЬТАТЫ WALK-FORWARD VALIDATION")
        print("=" * 80)
        print(results['summary'])
        
        # Сохраняем детальные результаты
        results_dir = Path('workspace/results/walk_forward')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame(results['folds'])
        results_path = results_dir / 'walk_forward_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"\n✓ Детальные результаты сохранены: {results_path}")
        
        # Выводим статистику по окнам
        print("\nСтатистика по окнам:")
        print(f"  Всего окон: {len(results['folds'])}")
        if 'accuracy_mean' in results['aggregated']:
            print(f"  Средняя точность: {results['aggregated']['accuracy_mean']:.2%}")
            print(f"  Стандартное отклонение: {results['aggregated']['accuracy_std']:.2%}")
            print(f"  Минимум: {results['aggregated']['accuracy_min']:.2%}")
            print(f"  Максимум: {results['aggregated']['accuracy_max']:.2%}")
        
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении walk-forward validation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 80)
    print("✓ WALK-FORWARD VALIDATION ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == '__main__':
    main()

