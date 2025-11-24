"""
Модуль для разделения данных на train/validation/test
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split

class DataSplitter:
    """
    Класс для разделения данных на обучающую, валидационную и тестовую выборки
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_state: int = 42,
                 temporal_split: bool = True):
        """
        Args:
            train_ratio: Доля обучающей выборки
            val_ratio: Доля валидационной выборки
            test_ratio: Доля тестовой выборки
            random_state: Seed для воспроизводимости
            temporal_split: Если True, разделение по времени (без перемешивания)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Сумма долей должна быть равна 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.temporal_split = temporal_split
    
    def split(self, df: pd.DataFrame, 
             target_column: str = 'signal_class',
             stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Разделяет данные на train/validation/test
        
        Args:
            df: DataFrame с данными
            target_column: Название колонки с целевой переменной
            stratify: Использовать ли стратификацию по классам
        
        Returns:
            Tuple (train_df, val_df, test_df)
        """
        if self.temporal_split:
            return self._temporal_split(df)
        else:
            return self._random_split(df, target_column, stratify)
    
    def _temporal_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Временное разделение (без перемешивания)
        """
        # Сортируем по времени
        df_sorted = df.sort_index()
        
        n_total = len(df_sorted)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)
        
        # Разделяем по времени
        train_df = df_sorted.iloc[:n_train]
        val_df = df_sorted.iloc[n_train:n_train + n_val]
        test_df = df_sorted.iloc[n_train + n_val:]
        
        print(f"Временное разделение:")
        print(f"  Train: {len(train_df)} образцов ({train_df.index[0]} - {train_df.index[-1]})")
        print(f"  Val:   {len(val_df)} образцов ({val_df.index[0]} - {val_df.index[-1]})")
        print(f"  Test:  {len(test_df)} образцов ({test_df.index[0]} - {test_df.index[-1]})")
        
        return train_df, val_df, test_df
    
    def _random_split(self, df: pd.DataFrame, 
                     target_column: str,
                     stratify: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Случайное разделение (с перемешиванием)
        """
        # Получаем фичи и целевую переменную
        feature_columns = [col for col in df.columns 
                          if col != target_column 
                          and not col.startswith('future_return')
                          and col != 'signal_class_name'
                          and col != 'max_future_return']
        
        X = df[feature_columns]
        y = df[target_column] if target_column in df.columns else None
        
        # Первое разделение: train и (val + test)
        stratify_param = y if stratify and y is not None else None
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.val_ratio + self.test_ratio),
            random_state=self.random_state,
            stratify=stratify_param
        )
        
        # Второе разделение: val и test
        if y_temp is not None:
            val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
            stratify_param = y_temp if stratify else None
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_size),
                random_state=self.random_state,
                stratify=stratify_param
            )
        else:
            val_size = self.val_ratio / (self.val_ratio + self.test_ratio)
            X_val, X_test = train_test_split(
                X_temp,
                test_size=(1 - val_size),
                random_state=self.random_state
            )
            y_val, y_test = None, None
        
        # Восстанавливаем DataFrame
        train_df = pd.DataFrame(X_train, index=X_train.index)
        if y_train is not None:
            train_df[target_column] = y_train
        
        val_df = pd.DataFrame(X_val, index=X_val.index)
        if y_val is not None:
            val_df[target_column] = y_val
        
        test_df = pd.DataFrame(X_test, index=X_test.index)
        if y_test is not None:
            test_df[target_column] = y_test
        
        # Добавляем остальные колонки
        other_columns = [col for col in df.columns 
                        if col not in feature_columns and col != target_column]
        for col in other_columns:
            if col in df.columns:
                train_df[col] = df.loc[train_df.index, col]
                val_df[col] = df.loc[val_df.index, col]
                test_df[col] = df.loc[test_df.index, col]
        
        print(f"Случайное разделение:")
        print(f"  Train: {len(train_df)} образцов")
        print(f"  Val:   {len(val_df)} образцов")
        print(f"  Test:  {len(test_df)} образцов")
        
        return train_df, val_df, test_df
    
    def get_class_distribution(self, df: pd.DataFrame, 
                              target_column: str = 'signal_class') -> pd.Series:
        """
        Возвращает распределение классов в датасете
        
        Args:
            df: DataFrame
            target_column: Название колонки с целевой переменной
        
        Returns:
            Series с распределением классов
        """
        if target_column not in df.columns:
            raise ValueError(f"Колонка '{target_column}' не найдена")
        
        return df[target_column].value_counts().sort_index()

