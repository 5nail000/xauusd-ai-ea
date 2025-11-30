"""
Быстрая проверка направления обучения модели
Использует упрощенные параметры для ускорения процесса проверки

Параметры для быстрой проверки:
- 1-2 месяца данных (вместо 6)
- Упрощенная модель (d_model=128, n_layers=2)
- Меньшая длина последовательности (30 вместо 60)
- Больший batch_size (64, если позволяет память)
- 20 эпох по умолчанию (вместо 100)
"""
import pandas as pd
import torch
import argparse
import sys
import os

# Добавляем корневую директорию в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import create_model, get_model_config
from models.data_loader import create_dataloaders
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from utils.feature_documentation import export_feature_documentation_for_model


def quick_train(training_months: int = 1,
                batch_size: int = 64,
                num_epochs: int = 20,
                early_stopping_patience: int = 5,
                sequence_length: int = 30,
                use_wandb: bool = False,
                wandb_project: str = 'xauusd-ai-ea',
                use_class_weights: bool = True,
                class_weight_method: str = 'balanced'):
    """
    Быстрое обучение модели для проверки направления
    
    Args:
        training_months: Количество месяцев данных (по умолчанию: 1)
        batch_size: Размер батча (по умолчанию: 64)
        num_epochs: Количество эпох (по умолчанию: 20)
        early_stopping_patience: Терпение для early stopping (по умолчанию: 5)
        sequence_length: Длина последовательности (по умолчанию: 30)
        use_wandb: Использовать Weights & Biases
        wandb_project: Название проекта в W&B
        use_class_weights: Использовать веса классов
        class_weight_method: Метод вычисления весов классов
    """
    print("=" * 80)
    print("БЫСТРАЯ ПРОВЕРКА НАПРАВЛЕНИЯ ОБУЧЕНИЯ")
    print("=" * 80)
    print(f"Параметры быстрой проверки:")
    print(f"  Месяцев данных: {training_months}")
    print(f"  Размер батча: {batch_size}")
    print(f"  Количество эпох: {num_epochs}")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Длина последовательности: {sequence_length}")
    print(f"  Модель: encoder (упрощенная)")
    print(f"  d_model: 128 (вместо 256)")
    print(f"  n_layers: 2 (вместо 4)")
    print(f"  n_heads: 4 (вместо 8)")
    print("=" * 80)
    
    # 1. Загрузка данных
    print("\n1. Загрузка данных...")
    try:
        train_df = pd.read_csv('workspace/prepared/features/gold_train.csv', index_col=0, parse_dates=True)
        val_df = pd.read_csv('workspace/prepared/features/gold_val.csv', index_col=0, parse_dates=True)
        test_df = pd.read_csv('workspace/prepared/features/gold_test.csv', index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"❌ Ошибка: Файл данных не найден: {e}")
        print("   Сначала запустите: python prepare_gold_data.py --months <N>")
        return None
    
    print(f"  Train: {len(train_df)} образцов (полная выборка)")
    print(f"  Val:   {len(val_df)} образцов (полная выборка)")
    print(f"  Test:  {len(test_df)} образцов (полная выборка)")
    
    # 1.1. Вычисляем веса классов ДО фильтрации (от полной выборки)
    # Это важно: веса должны соответствовать полному обучению, чтобы результаты были сопоставимы
    class_weights = None
    if use_class_weights:
        from models.data_loader import compute_class_weights
        print("\n1.1. Вычисление весов классов от полной выборки...")
        print("     (Веса вычисляются от полной выборки для сопоставимости с полным обучением)")
        class_weights = compute_class_weights(
            train_df,  # Используем полную выборку для весов
            target_column='signal_class',
            method=class_weight_method
        )
    
    # 1.2. Ограничиваем данные до указанного количества месяцев (для быстрой проверки)
    if training_months > 0:
        from datetime import timedelta
        print(f"\n1.2. Ограничение данных до последних {training_months} месяцев...")
        
        # Берем последние N месяцев из train_df
        end_date = train_df.index.max()
        start_date = end_date - timedelta(days=training_months * 30)  # Примерно 30 дней в месяце
        
        train_df_filtered = train_df[train_df.index >= start_date].copy()
        val_df_filtered = val_df[val_df.index >= start_date].copy()
        test_df_filtered = test_df[test_df.index >= start_date].copy()
        
        print(f"  Период: {start_date.date()} - {end_date.date()}")
        print(f"  Train: {len(train_df_filtered)} образцов (было {len(train_df)})")
        print(f"  Val:   {len(val_df_filtered)} образцов (было {len(val_df)})")
        print(f"  Test:  {len(test_df_filtered)} образцов (было {len(test_df)})")
        
        # Используем отфильтрованные данные для обучения
        train_df = train_df_filtered
        val_df = val_df_filtered
        test_df = test_df_filtered
    
    # 2. Создание DataLoader'ов
    print("\n2. Создание последовательностей...")
    # exclude_columns будет загружен автоматически из excluded_features.txt внутри create_dataloaders
    train_loader, val_loader, test_loader, seq_gen = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sequence_length=sequence_length,
        batch_size=batch_size,
        target_column='signal_class'
    )
    
    num_features = train_loader.dataset.sequences.shape[2]
    print(f"  Размерность фичей: {num_features}")
    
    # Сохраняем scaler с метаданными
    scaler_path = 'workspace/prepared/scalers/feature_scaler_quick_test.pkl'
    
    metadata = {
        'training_months': training_months,
        'model_type': 'encoder',
        'num_features': num_features,
        'sequence_length': sequence_length,
        'target_column': 'signal_class',
        'quick_test': True,
        'preparation_config': {
            'remove_correlated_features': False,
            'correlation_threshold': 0.95
        }
    }
    
    seq_gen.save_scaler(scaler_path, metadata=metadata)
    print(f"  Scaler сохранен: {scaler_path}")
    print(f"  Метаданные: {training_months} месяцев, {num_features} фичей")
    
    # 3. Создание упрощенной модели
    print("\n3. Создание упрощенной модели...")
    model_type = 'encoder'
    
    config = get_model_config(
        model_type='encoder',
        num_features=num_features,
        num_classes=5,
        sequence_length=sequence_length,
        d_model=128,      # Уменьшено с 256
        n_layers=2,       # Уменьшено с 4
        n_heads=4,        # Уменьшено с 8
        dropout=0.1
    )
    
    config.learning_rate = 1e-4
    config.batch_size = batch_size
    config.num_epochs = num_epochs
    config.early_stopping_patience = early_stopping_patience
    config.training_data_months = training_months
    
    model = create_model(config)
    print(f"  Тип модели: {model_type}")
    print(f"  Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  sequence_length: {config.sequence_length}")
    
    # 4. Обучение
    print("\n4. Обучение модели...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Устройство: {device}")
    
    trainer = ModelTrainer(
        model=model,
        device=device,
        learning_rate=config.learning_rate,
        weight_decay=1e-5,
        scheduler_type='cosine',
        model_config=config,
        model_type=model_type,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        use_class_weights=use_class_weights,
        class_weights=class_weights
    )
    
    checkpoint_path = 'workspace/models/checkpoints/encoder_model_quick_test.pth'
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        checkpoint_path=checkpoint_path,
        save_history=True
    )
    
    # 5. Оценка модели
    print("\n5. Оценка модели...")
    evaluator = ModelEvaluator(model, device)
    
    # Оценка на всех выборках
    results = evaluator.evaluate_all_splits(train_loader, val_loader, test_loader)
    
    # Classification report
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT (TEST) - QUICK TEST")
    print("=" * 80)
    evaluator.print_classification_report(test_loader)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        test_loader,
        save_path='workspace/models/metrics/confusion_matrix_quick_test.png',
        show=False
    )
    
    # Распределение классов
    print("\nРаспределение классов:")
    print(evaluator.get_class_distribution(test_loader))
    
    # 6. Экспорт документации по фичам
    print("\n6. Экспорт документации по фичам...")
    try:
        doc_path = export_feature_documentation_for_model(
            model_type=model_type,
            feature_columns=seq_gen.feature_columns,
            scaler_path=scaler_path
        )
        print(f"  ✓ Документация сохранена: {doc_path}.json и {doc_path}.md")
    except Exception as e:
        print(f"  ⚠️  Ошибка при экспорте документации: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ БЫСТРАЯ ПРОВЕРКА ЗАВЕРШЕНА!")
    print("=" * 80)
    print(f"Модель сохранена: {checkpoint_path}")
    print(f"Scaler сохранен: {scaler_path}")
    print(f"История обучения: workspace/models/metrics/encoder_model_quick_test_history.csv")
    print(f"Графики обучения: workspace/models/metrics/encoder_model_quick_test_training_curves.png")
    print("=" * 80)
    
    # Вывод рекомендаций
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ ПО РЕЗУЛЬТАТАМ")
    print("=" * 80)
    test_results = results.get('test', {})
    accuracy = test_results.get('accuracy', 0)
    val_results = results.get('val', {})
    val_accuracy = val_results.get('accuracy', 0)
    train_results = results.get('train', {})
    train_accuracy = train_results.get('accuracy', 0)
    
    gap = val_accuracy - train_accuracy
    
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Val Accuracy:   {val_accuracy:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Gap (Val - Train): {gap:.4f}")
    
    if gap < -0.05:
        print("\n⚠️  Обнаружено переобучение (gap < -5%)")
        print("   Рекомендации:")
        print("   - Увеличить dropout (0.1 → 0.2)")
        print("   - Добавить weight_decay (1e-5 → 1e-4)")
        print("   - Уменьшить сложность модели")
    elif gap > 0.05:
        print("\n✓ Переобучение не обнаружено")
        print("   Модель показывает стабильные результаты")
    else:
        print("\n✓ Gap в пределах нормы")
    
    if accuracy < 0.3:
        print("\n⚠️  Низкая точность (< 30%)")
        print("   Рекомендации:")
        print("   - Проверить качество данных")
        print("   - Увеличить количество эпох")
        print("   - Проверить баланс классов")
    elif accuracy > 0.5:
        print("\n✓ Хорошая точность (> 50%)")
        print("   Можно переходить к полному обучению")
    
    print("=" * 80)
    
    return results


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description='Быстрая проверка направления обучения модели',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python utils/quick_test_training.py                    # 1 месяц, все параметры по умолчанию
  python utils/quick_test_training.py --months 2         # 2 месяца данных
  python utils/quick_test_training.py --batch-size 32    # Меньший batch (если не хватает памяти)
  python utils/quick_test_training.py --epochs 15        # 15 эпох вместо 20
  python utils/quick_test_training.py --sequence-length 40  # Длина последовательности 40
        """
    )
    
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=1,
        help='Количество месяцев данных (по умолчанию: 1)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Размер батча (по умолчанию: 64, уменьшите до 32 если не хватает памяти)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Количество эпох (по умолчанию: 20)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=5,
        help='Терпение для early stopping (по умолчанию: 5)'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=30,
        help='Длина последовательности (по умолчанию: 30)'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Использовать Weights & Biases для логирования'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='xauusd-ai-ea-quick-test',
        help='Название проекта в W&B (по умолчанию: xauusd-ai-ea-quick-test)'
    )
    
    parser.add_argument(
        '--no-class-weights',
        action='store_true',
        help='НЕ использовать веса классов (по умолчанию веса включены)'
    )
    
    parser.add_argument(
        '--class-weight-method',
        type=str,
        default='balanced',
        choices=['balanced', 'inverse', 'sqrt'],
        help='Метод вычисления весов классов (по умолчанию: balanced)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("БЫСТРАЯ ПРОВЕРКА НАПРАВЛЕНИЯ ОБУЧЕНИЯ")
    print("=" * 80)
    print(f"Параметры:")
    print(f"  Месяцев данных: {args.months}")
    print(f"  Размер батча: {args.batch_size}")
    print(f"  Количество эпох: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Длина последовательности: {args.sequence_length}")
    print(f"  Weights & Biases: {'Включено' if args.use_wandb else 'Выключено'}")
    if args.use_wandb:
        print(f"  W&B проект: {args.wandb_project}")
    print("=" * 80)
    
    try:
        results = quick_train(
            training_months=args.months,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            early_stopping_patience=args.patience,
            sequence_length=args.sequence_length,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            use_class_weights=not args.no_class_weights,
            class_weight_method=args.class_weight_method
        )
        
        if results:
            print("\n" + "=" * 80)
            print("ИТОГОВАЯ СВОДКА")
            print("=" * 80)
            test_results = results.get('test', {})
            print(f"\nTest Accuracy:  {test_results.get('accuracy', 0):.4f}")
            print(f"Test Precision: {test_results.get('precision_weighted', 0):.4f}")
            print(f"Test Recall:    {test_results.get('recall_weighted', 0):.4f}")
            print(f"Test F1-Score:  {test_results.get('f1_weighted', 0):.4f}")
            print("=" * 80)
            
    except Exception as e:
        print(f"\n❌ Ошибка при быстрой проверке: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

