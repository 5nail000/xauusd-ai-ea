"""
Скрипт для обучения Transformer модели
"""
import pandas as pd
import torch
from models.model_factory import create_model, get_model_config
from models.data_loader import create_dataloaders
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from config.model_config import TransformerConfig

def main():
    """
    Обучение Transformer модели для классификации торговых сигналов
    """
    print("=" * 60)
    print("Обучение Transformer модели")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    train_df = pd.read_csv('data/gold_train.csv', index_col=0, parse_dates=True)
    val_df = pd.read_csv('data/gold_val.csv', index_col=0, parse_dates=True)
    test_df = pd.read_csv('data/gold_test.csv', index_col=0, parse_dates=True)
    
    print(f"  Train: {len(train_df)} образцов")
    print(f"  Val:   {len(val_df)} образцов")
    print(f"  Test:  {len(test_df)} образцов")
    
    # 2. Создание DataLoader'ов
    print("\n2. Создание последовательностей...")
    train_loader, val_loader, test_loader, seq_gen = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sequence_length=60,
        batch_size=32,
        target_column='signal_class'
    )
    
    # Сохраняем scaler для использования в продакшене
    seq_gen.save_scaler('models/feature_scaler.pkl')
    print("  Scaler сохранен: models/feature_scaler.pkl")
    
    # 3. Создание модели
    print("\n3. Создание модели...")
    model_type = 'encoder'  # или 'timeseries' для продвинутой модели
    
    config = get_model_config(
        model_type=model_type,
        num_features=train_loader.dataset.sequences.shape[2],
        num_classes=3,
        sequence_length=60,
        d_model=256,
        n_layers=4,
        n_heads=8,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=32,
        num_epochs=100,
        early_stopping_patience=10,
        training_data_months=6
    )
    
    model = create_model(config)
    print(f"  Тип модели: {model_type}")
    print(f"  Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Обучение
    print("\n4. Обучение модели...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=1e-5,
        scheduler_type='cosine'
    )
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        early_stopping_patience=config.early_stopping_patience,
        checkpoint_path=f'models/checkpoints/{model_type}_model.pth',
        save_history=True
    )
    
    # 5. Оценка модели
    print("\n5. Оценка модели...")
    evaluator = ModelEvaluator(model, device)
    
    # Оценка на всех выборках
    results = evaluator.evaluate_all_splits(train_loader, val_loader, test_loader)
    
    # Classification report
    print("\n" + "=" * 60)
    print("Classification Report (Test)")
    print("=" * 60)
    evaluator.print_classification_report(test_loader)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        test_loader,
        save_path=f'models/confusion_matrix_{model_type}.png'
    )
    
    # Распределение классов
    print("\nРаспределение классов:")
    print(evaluator.get_class_distribution(test_loader))
    
    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print(f"Модель сохранена: models/checkpoints/{model_type}_model.pth")
    print("=" * 60)

if __name__ == '__main__':
    main()

