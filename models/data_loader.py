"""
DataLoader для загрузки последовательностей для Transformer
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import pickle
import os

class TimeSeriesDataset(Dataset):
    """
    Dataset для временных рядов с последовательностями
    """
    
    def __init__(self, 
                 sequences: np.ndarray,
                 targets: np.ndarray,
                 feature_names: Optional[List[str]] = None):
        """
        Args:
            sequences: Массив последовательностей [n_samples, seq_len, n_features]
            targets: Массив целевых переменных [n_samples]
            feature_names: Список названий фичей (опционально)
        """
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)
        self.feature_names = feature_names
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]

class SequenceGenerator:
    """
    Генератор последовательностей из DataFrame для Transformer
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 target_column: str = 'signal_class',
                 exclude_columns: Optional[List[str]] = None):
        """
        Args:
            sequence_length: Длина последовательности
            target_column: Название колонки с целевой переменной
            exclude_columns: Колонки для исключения из фичей
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.exclude_columns = exclude_columns or []
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Получает список колонок с фичами"""
        exclude_patterns = [
            'target', 'label', 'direction', 'future_return', 
            'future_volatility', 'signal_class_name', 'max_future_return'
        ]
        
        feature_columns = [
            col for col in df.columns
            if col != self.target_column
            and not any(pattern in col.lower() for pattern in exclude_patterns)
            and col not in self.exclude_columns
        ]
        
        return feature_columns
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Создает последовательности из DataFrame
        
        Args:
            df: DataFrame с фичами и целевой переменной
        
        Returns:
            Tuple (sequences, targets)
            - sequences: [n_samples, seq_len, n_features]
            - targets: [n_samples]
        """
        if self.feature_columns is None:
            self.feature_columns = self._get_feature_columns(df)
        
        # Удаляем строки с NaN в целевой переменной
        df_clean = df.dropna(subset=[self.target_column])
        
        # Выбираем только фичи
        feature_data = df_clean[self.feature_columns].values
        
        # Нормализация (если еще не обучен scaler)
        if not self.is_fitted:
            feature_data = self.scaler.fit_transform(feature_data)
            self.is_fitted = True
        else:
            feature_data = self.scaler.transform(feature_data)
        
        # Создаем последовательности
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(feature_data)):
            sequence = feature_data[i - self.sequence_length:i]
            target = df_clean[self.target_column].iloc[i]
            
            sequences.append(sequence)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def fit_scaler(self, df: pd.DataFrame):
        """Обучает scaler на данных"""
        if self.feature_columns is None:
            self.feature_columns = self._get_feature_columns(df)
        
        feature_data = df[self.feature_columns].dropna().values
        self.scaler.fit(feature_data)
        self.is_fitted = True
    
    def save_scaler(self, filepath: str):
        """Сохраняет scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'target_column': self.target_column
            }, f)
    
    def load_scaler(self, filepath: str):
        """Загружает scaler"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.feature_columns = data['feature_columns']
            self.sequence_length = data['sequence_length']
            self.target_column = data['target_column']
            self.is_fitted = True

def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       sequence_length: int = 60,
                       batch_size: int = 32,
                       target_column: str = 'signal_class',
                       num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader, SequenceGenerator]:
    """
    Создает DataLoader'ы для train/val/test
    
    Args:
        train_df: Обучающая выборка
        val_df: Валидационная выборка
        test_df: Тестовая выборка
        sequence_length: Длина последовательности
        batch_size: Размер батча
        target_column: Название колонки с целевой переменной
        num_workers: Количество воркеров для DataLoader
    
    Returns:
        Tuple (train_loader, val_loader, test_loader, sequence_generator)
    """
    # Создаем генератор последовательностей
    seq_gen = SequenceGenerator(
        sequence_length=sequence_length,
        target_column=target_column
    )
    
    # Обучаем scaler на train данных
    seq_gen.fit_scaler(train_df)
    
    # Создаем последовательности
    train_sequences, train_targets = seq_gen.create_sequences(train_df)
    val_sequences, val_targets = seq_gen.create_sequences(val_df)
    test_sequences, test_targets = seq_gen.create_sequences(test_df)
    
    print(f"Создано последовательностей:")
    print(f"  Train: {len(train_sequences)}")
    print(f"  Val:   {len(val_sequences)}")
    print(f"  Test:  {len(test_sequences)}")
    print(f"  Размерность фичей: {train_sequences.shape[2]}")
    
    # Создаем Dataset'ы
    train_dataset = TimeSeriesDataset(train_sequences, train_targets)
    val_dataset = TimeSeriesDataset(val_sequences, val_targets)
    test_dataset = TimeSeriesDataset(test_sequences, test_targets)
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, seq_gen

