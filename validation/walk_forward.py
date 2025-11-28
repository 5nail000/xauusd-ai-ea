"""
Walk-Forward Validation для временных рядов

Идея: симулировать реальную торговлю, где модель обучается на исторических данных
и тестируется на будущих данных, с движущимся окном.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
from datetime import datetime, timedelta


class WalkForwardValidator:
    """
    Walk-Forward Validation для временных рядов
    
    Идея:
    - Разбиваем данные на несколько "окон"
    - Для каждого окна: обучаем на train, валидируем на val, тестируем на test
    - Агрегируем результаты по всем окнам
    
    Это более реалистичный способ оценки модели, чем простое train/val/test разделение,
    так как симулирует реальную торговлю, где модель обучается на прошлом и тестируется на будущем.
    """
    
    def __init__(self,
                 train_size: int = 100,  # Размер обучающего окна (дни или образцы)
                 val_size: int = 20,     # Размер валидационного окна
                 test_size: int = 20,    # Размер тестового окна
                 step_size: int = 10,    # Шаг сдвига окна
                 use_days: bool = False):  # Использовать дни или образцы
        """
        Args:
            train_size: Размер обучающего окна (в днях если use_days=True, иначе в образцах)
            val_size: Размер валидационного окна
            test_size: Размер тестового окна
            step_size: Шаг сдвига окна (как часто создавать новое окно)
            use_days: Если True, размеры в днях (требует DatetimeIndex), иначе в образцах
        """
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.step_size = step_size
        self.use_days = use_days
    
    def create_folds(self, df: pd.DataFrame) -> List[Dict]:
        """
        Создает список "окон" для walk-forward validation
        
        Args:
            df: DataFrame с временным индексом
        
        Returns:
            Список словарей с индексами train/val/test для каждого окна
        """
        df_sorted = df.sort_index()
        
        if not isinstance(df_sorted.index, pd.DatetimeIndex) and self.use_days:
            raise ValueError("Для use_days=True требуется DatetimeIndex")
        
        folds = []
        start_idx = 0
        
        if self.use_days:
            # Работаем с днями
            start_date = df_sorted.index[0]
            end_date = df_sorted.index[-1]
            
            current_date = start_date
            while True:
                train_end_date = current_date + timedelta(days=self.train_size)
                val_end_date = train_end_date + timedelta(days=self.val_size)
                test_end_date = val_end_date + timedelta(days=self.test_size)
                
                if test_end_date > end_date:
                    break
                
                # Находим индексы для дат
                train_mask = (df_sorted.index >= current_date) & (df_sorted.index < train_end_date)
                val_mask = (df_sorted.index >= train_end_date) & (df_sorted.index < val_end_date)
                test_mask = (df_sorted.index >= val_end_date) & (df_sorted.index < test_end_date)
                
                train_indices = df_sorted.index[train_mask]
                val_indices = df_sorted.index[val_mask]
                test_indices = df_sorted.index[test_mask]
                
                if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
                    current_date += timedelta(days=self.step_size)
                    continue
                
                fold = {
                    'fold_id': len(folds) + 1,
                    'train_start': train_indices[0],
                    'train_end': train_indices[-1],
                    'val_start': val_indices[0],
                    'val_end': val_indices[-1],
                    'test_start': test_indices[0],
                    'test_end': test_indices[-1],
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'test_indices': test_indices
                }
                
                folds.append(fold)
                current_date += timedelta(days=self.step_size)
        else:
            # Работаем с образцами
            total_samples = len(df_sorted)
            
            while start_idx + self.train_size + self.val_size + self.test_size <= total_samples:
                # Определяем границы окон
                train_end = start_idx + self.train_size
                val_end = train_end + self.val_size
                test_end = val_end + self.test_size
                
                fold = {
                    'fold_id': len(folds) + 1,
                    'train_start': df_sorted.index[start_idx],
                    'train_end': df_sorted.index[train_end - 1],
                    'val_start': df_sorted.index[train_end],
                    'val_end': df_sorted.index[val_end - 1],
                    'test_start': df_sorted.index[val_end],
                    'test_end': df_sorted.index[test_end - 1],
                    'train_indices': df_sorted.index[start_idx:train_end],
                    'val_indices': df_sorted.index[train_end:val_end],
                    'test_indices': df_sorted.index[val_end:test_end]
                }
                
                folds.append(fold)
                
                # Сдвигаем окно
                start_idx += self.step_size
        
        return folds
    
    def validate(self, 
                 df: pd.DataFrame,
                 train_function: Callable,  # Функция обучения модели
                 evaluate_function: Callable,  # Функция оценки модели
                 verbose: bool = True,
                 **train_kwargs) -> Dict:
        """
        Выполняет walk-forward validation
        
        Args:
            df: DataFrame с данными
            train_function: Функция для обучения модели
                Сигнатура: model = train_function(train_df, val_df, **kwargs)
            evaluate_function: Функция для оценки модели
                Сигнатура: metrics = evaluate_function(model, test_df, **kwargs)
            verbose: Выводить ли подробную информацию
            **train_kwargs: Дополнительные параметры для обучения
        
        Returns:
            Словарь с результатами валидации:
            {
                'folds': [список результатов по каждому окну],
                'aggregated': {агрегированные метрики},
                'summary': 'текстовая сводка'
            }
        """
        folds = self.create_folds(df)
        
        if len(folds) == 0:
            raise ValueError("Не удалось создать ни одного окна. Проверьте размеры окон и данные.")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"WALK-FORWARD VALIDATION")
            print(f"{'='*60}")
            print(f"Создано {len(folds)} окон для walk-forward validation")
            print(f"Размеры: Train={self.train_size}, Val={self.val_size}, Test={self.test_size}")
            print(f"Шаг: {self.step_size}")
            if self.use_days:
                print(f"Единицы: дни")
            else:
                print(f"Единицы: образцы")
            print(f"{'='*60}\n")
        
        all_results = []
        
        for i, fold in enumerate(folds):
            if verbose:
                print(f"Окно {i+1}/{len(folds)}:")
                print(f"  Train: {fold['train_start']} - {fold['train_end']} ({len(fold['train_indices'])} образцов)")
                print(f"  Val:   {fold['val_start']} - {fold['val_end']} ({len(fold['val_indices'])} образцов)")
                print(f"  Test:  {fold['test_start']} - {fold['test_end']} ({len(fold['test_indices'])} образцов)")
            
            # Разделяем данные
            train_df = df.loc[fold['train_indices']]
            val_df = df.loc[fold['val_indices']]
            test_df = df.loc[fold['test_indices']]
            
            # Обучаем модель
            if verbose:
                print("  Обучение модели...")
            try:
                model = train_function(train_df, val_df, **train_kwargs)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Ошибка при обучении: {e}")
                continue
            
            # Оцениваем на тестовом наборе
            if verbose:
                print("  Оценка на тестовом наборе...")
            try:
                metrics = evaluate_function(model, test_df, **train_kwargs)
            except Exception as e:
                if verbose:
                    print(f"  ❌ Ошибка при оценке: {e}")
                continue
            
            # Сохраняем результаты
            result = {
                'fold_id': fold['fold_id'],
                'train_period': (fold['train_start'], fold['train_end']),
                'val_period': (fold['val_start'], fold['val_end']),
                'test_period': (fold['test_start'], fold['test_end']),
                'train_samples': len(fold['train_indices']),
                'val_samples': len(fold['val_indices']),
                'test_samples': len(fold['test_indices']),
                **metrics
            }
            
            all_results.append(result)
            
            if verbose:
                # Выводим ключевые метрики
                key_metrics = ['accuracy', 'f1', 'precision', 'recall']
                metrics_str = ", ".join([f"{k}={metrics.get(k, 0):.4f}" 
                                        for k in key_metrics if k in metrics])
                print(f"  ✓ Результаты: {metrics_str}\n")
        
        if len(all_results) == 0:
            raise ValueError("Не удалось получить результаты ни для одного окна.")
        
        # Агрегируем результаты
        aggregated = self._aggregate_results(all_results)
        
        return {
            'folds': all_results,
            'aggregated': aggregated,
            'summary': self._create_summary(aggregated, len(all_results))
        }
    
    def _aggregate_results(self, results: List[Dict]) -> Dict:
        """Агрегирует результаты по всем окнам"""
        # Собираем все метрики из результатов
        all_metrics = set()
        for r in results:
            all_metrics.update([k for k in r.keys() 
                              if k not in ['fold_id', 'train_period', 'val_period', 'test_period',
                                          'train_samples', 'val_samples', 'test_samples']])
        
        aggregated = {}
        for metric in all_metrics:
            values = [r[metric] for r in results if metric in r and isinstance(r[metric], (int, float))]
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        return aggregated
    
    def _create_summary(self, aggregated: Dict, num_folds: int) -> str:
        """Создает текстовую сводку результатов"""
        summary = "\n" + "=" * 60 + "\n"
        summary += "WALK-FORWARD VALIDATION SUMMARY\n"
        summary += "=" * 60 + "\n\n"
        summary += f"Количество окон: {num_folds}\n"
        summary += f"Размеры окон: Train={self.train_size}, Val={self.val_size}, Test={self.test_size}\n"
        summary += f"Шаг: {self.step_size}\n\n"
        
        # Ключевые метрики
        key_metrics = ['accuracy', 'f1', 'precision', 'recall', 'loss']
        
        for metric in key_metrics:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            min_key = f'{metric}_min'
            max_key = f'{metric}_max'
            
            if mean_key in aggregated:
                mean_val = aggregated[mean_key]
                std_val = aggregated[std_key]
                min_val = aggregated[min_key]
                max_val = aggregated[max_key]
                
                if metric == 'loss':
                    summary += f"{metric.capitalize():12s}: {mean_val:.4f} (±{std_val:.4f}, "
                    summary += f"range: {min_val:.4f} - {max_val:.4f})\n"
                else:
                    summary += f"{metric.capitalize():12s}: {mean_val:.2%} (±{std_val:.2%}, "
                    summary += f"range: {min_val:.2%} - {max_val:.2%})\n"
        
        summary += "\n" + "=" * 60 + "\n"
        
        return summary

