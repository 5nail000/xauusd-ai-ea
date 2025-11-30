"""
Скрипт для обучения всех моделей на заданном количестве месяцев данных
"""
import pandas as pd
import torch
import argparse
from models.model_factory import create_model, get_model_config
from models.data_loader import create_dataloaders
from models.trainer import ModelTrainer
from models.evaluator import ModelEvaluator
from utils.feature_documentation import export_feature_documentation_for_model

def train_model_type(model_type: str, training_months: int = 12, batch_size: int = 32, 
                     num_epochs: int = 100, early_stopping_patience: int = 10,
                     use_wandb: bool = False, wandb_project: str = 'xauusd-ai-ea',
                     use_class_weights: bool = True, class_weight_method: str = 'balanced'):
    """Обучает модель указанного типа"""
    print("=" * 80)
    print(f"ОБУЧЕНИЕ {model_type.upper()} МОДЕЛИ НА {training_months} МЕСЯЦАХ")
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
    
    print(f"  Train: {len(train_df)} образцов")
    print(f"  Val:   {len(val_df)} образцов")
    print(f"  Test:  {len(test_df)} образцов")
    
    # 2. Создание DataLoader'ов
    print("\n2. Создание последовательностей...")
    # exclude_columns будет загружен автоматически из excluded_features.txt внутри create_dataloaders
    train_loader, val_loader, test_loader, seq_gen = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        sequence_length=60,
        batch_size=batch_size,
        target_column='signal_class'
    )
    
    num_features = train_loader.dataset.sequences.shape[2]
    print(f"  Размерность фичей: {num_features}")
    
    # 2.1. Вычисляем веса классов (если нужно)
    class_weights = None
    if use_class_weights:
        from models.data_loader import compute_class_weights
        class_weights = compute_class_weights(
            train_df, 
            target_column='signal_class',
            method=class_weight_method
        )
    
    # Сохраняем scaler для каждого типа модели с метаданными
    scaler_path = f'workspace/prepared/scalers/feature_scaler_{model_type}.pkl'
    
    # Создаем метаданные о подготовке данных
    metadata = {
        'training_months': training_months,
        'model_type': model_type,
        'num_features': num_features,
        'sequence_length': 60,
        'target_column': 'signal_class',
        'preparation_config': {
            'remove_correlated_features': False,  # Будет обновлено из конфига
            'correlation_threshold': 0.95
        }
    }
    
    seq_gen.save_scaler(scaler_path, metadata=metadata)
    print(f"  Scaler сохранен: {scaler_path}")
    print(f"  Метаданные: {training_months} месяцев, {num_features} фичей")
    
    # 3. Создание модели
    print("\n3. Создание модели...")
    
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
    elif model_type == 'timeseries':
        config = get_model_config(
            model_type='timeseries',
            num_features=num_features,
            num_classes=5,
            sequence_length=60,
            d_model=256,
            n_layers=6,
            n_heads=8,
            dropout=0.1,
            use_temporal_encoding=True,
            use_patch_embedding=False
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    config.learning_rate = 1e-4
    config.batch_size = batch_size
    config.num_epochs = num_epochs
    config.early_stopping_patience = early_stopping_patience
    config.training_data_months = training_months
    
    model = create_model(config)
    print(f"  Тип модели: {model_type}")
    print(f"  Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    checkpoint_path = f'workspace/models/checkpoints/{model_type}_model.pth'
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
    print(f"CLASSIFICATION REPORT (TEST) - {model_type.upper()}")
    print("=" * 80)
    evaluator.print_classification_report(test_loader)
    
    # Confusion matrix
    evaluator.plot_confusion_matrix(
        test_loader,
        save_path=f'workspace/models/metrics/confusion_matrix_{model_type}.png',
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
    print(f"✓ ОБУЧЕНИЕ {model_type.upper()} МОДЕЛИ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"Модель сохранена: {checkpoint_path}")
    print(f"Scaler сохранен: {scaler_path}")
    print(f"Документация по фичам: {doc_path if 'doc_path' in locals() else 'N/A'}.json/.md")
    print(f"История обучения: workspace/models/metrics/{model_type}_model_history.csv")
    print(f"Графики обучения: workspace/models/metrics/{model_type}_model_training_curves.png")
    print("=" * 80)
    
    return results

def main():
    """Обучает обе модели последовательно"""
    parser = argparse.ArgumentParser(
        description='Обучение всех моделей (encoder и timeseries)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python train_all_models.py                    # 12 месяцев, все параметры по умолчанию
  python train_all_models.py --months 6         # 6 месяцев
  python train_all_models.py --encoder-only      # Только encoder модель
  python train_all_models.py --timeseries-only  # Только timeseries модель
  python train_all_models.py --epochs 50        # 50 эпох вместо 100
        """
    )
    
    parser.add_argument(
        '-m', '--months',
        type=int,
        default=12,
        help='Количество месяцев данных (по умолчанию: 12)'
    )
    
    parser.add_argument(
        '--encoder-only',
        action='store_true',
        help='Обучить только encoder модель'
    )
    
    parser.add_argument(
        '--timeseries-only',
        action='store_true',
        help='Обучить только timeseries модель'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Размер батча (по умолчанию: 32)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Количество эпох (по умолчанию: 100)'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Терпение для early stopping (по умолчанию: 10)'
    )
    
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Использовать Weights & Biases для логирования'
    )
    
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='xauusd-ai-ea',
        help='Название проекта в W&B (по умолчанию: xauusd-ai-ea)'
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
    print("ОБУЧЕНИЕ ВСЕХ МОДЕЛЕЙ")
    print("=" * 80)
    print(f"Параметры:")
    print(f"  Месяцев данных: {args.months}")
    print(f"  Размер батча: {args.batch_size}")
    print(f"  Количество эпох: {args.epochs}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Weights & Biases: {'Включено' if args.use_wandb else 'Выключено'}")
    if args.use_wandb:
        print(f"  W&B проект: {args.wandb_project}")
    print("=" * 80)
    
    models_to_train = []
    
    if args.encoder_only:
        models_to_train = ['encoder']
    elif args.timeseries_only:
        models_to_train = ['timeseries']
    else:
        models_to_train = ['encoder', 'timeseries']
    
    print(f"\nБудут обучены модели: {', '.join(models_to_train)}")
    print("=" * 80)
    
    all_results = {}
    
    for i, model_type in enumerate(models_to_train, 1):
        if len(models_to_train) > 1:
            print(f"\n\n{'='*80}")
            print(f"МОДЕЛЬ {i}/{len(models_to_train)}: {model_type.upper()}")
            print(f"{'='*80}")
        
        try:
            results = train_model_type(
                model_type=model_type,
                training_months=args.months,
                batch_size=args.batch_size,
                num_epochs=args.epochs,
                early_stopping_patience=args.patience,
                use_wandb=args.use_wandb,
                wandb_project=args.wandb_project,
                use_class_weights=not args.no_class_weights,
                class_weight_method=args.class_weight_method
            )
            all_results[model_type] = results
        except Exception as e:
            print(f"\n❌ Ошибка при обучении {model_type} модели: {e}")
            import traceback
            traceback.print_exc()
            if len(models_to_train) > 1:
                print(f"\n⚠️  Продолжаем с следующей моделью...")
            else:
                return
    
    # Итоговая сводка
    print("\n\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 80)
    
    for model_type in models_to_train:
        if model_type in all_results and all_results[model_type]:
            test_results = all_results[model_type].get('test', {})
            print(f"\n{model_type.upper()} модель:")
            print(f"  Accuracy:  {test_results.get('accuracy', 0):.4f}")
            print(f"  Precision: {test_results.get('precision_weighted', 0):.4f}")
            print(f"  Recall:    {test_results.get('recall_weighted', 0):.4f}")
            print(f"  F1-Score:  {test_results.get('f1_weighted', 0):.4f}")
    
    print("\n" + "=" * 80)
    print("ВСЕ МОДЕЛИ ОБУЧЕНЫ!")
    print("=" * 80)
    print("\nСохраненные файлы:")
    for model_type in models_to_train:
        print(f"\n{model_type.upper()} модель:")
        print(f"  - workspace/models/checkpoints/{model_type}_model.pth")
        print(f"  - workspace/prepared/scalers/feature_scaler_{model_type}.pkl")
        print(f"  - workspace/models/checkpoints/{model_type}_model_features_documentation.json")
        print(f"  - workspace/models/checkpoints/{model_type}_model_features_documentation.md")
        print(f"  - workspace/models/metrics/{model_type}_model_history.csv")
        print(f"  - workspace/models/metrics/{model_type}_model_history.pkl")
        print(f"  - workspace/models/metrics/{model_type}_model_training_curves.png")
        print(f"  - workspace/models/metrics/confusion_matrix_{model_type}.png")
    print("=" * 80)

if __name__ == '__main__':
    main()

