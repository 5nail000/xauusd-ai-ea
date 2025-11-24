"""
DataLoader для загрузки последовательностей для Transformer
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler
import pickle
import os

class TimeSeriesDataset(Dataset):
    """
    Dataset для временных рядов с последовательностями
    Использует ленивую загрузку для экономии памяти
    """
    
    def __init__(self, 
                 sequences: np.ndarray,
                 targets: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 lazy_load: bool = True):
        """
        Args:
            sequences: Массив последовательностей [n_samples, seq_len, n_features]
            targets: Массив целевых переменных [n_samples]
            feature_names: Список названий фичей (опционально)
            lazy_load: Если True, не загружает все данные в память сразу
        """
        self.sequences = sequences  # Храним как numpy array для экономии памяти
        self.targets = targets
        self.feature_names = feature_names
        self.lazy_load = lazy_load
        
        # Если lazy_load=False, загружаем все в память (старое поведение)
        if not lazy_load:
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.LongTensor(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.lazy_load:
            # Конвертируем в tensor только при запросе
            sequence = torch.FloatTensor(self.sequences[idx])
            target = torch.LongTensor([self.targets[idx]])[0]
            return sequence, target
        else:
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
        
        # Выбираем только фичи и заполняем пропуски, чтобы не терять строки
        # Важно: при применении (is_fitted=True) используем только сохраненные фичи в правильном порядке
        if self.is_fitted and self.feature_columns is not None:
            # Проверяем наличие всех сохраненных фичей
            missing_features = set(self.feature_columns) - set(df_clean.columns)
            if missing_features:
                raise ValueError(
                    f"❌ ОШИБКА: Отсутствуют фичи, необходимые для модели:\n"
                    f"   {sorted(list(missing_features))[:10]}{'...' if len(missing_features) > 10 else ''}\n"
                    f"   Всего отсутствует: {len(missing_features)} из {len(self.feature_columns)}\n"
                    f"   Убедитесь, что данные подготовлены с теми же настройками, что и при обучении."
                )
            # Используем фичи в том же порядке, что и при обучении
            feature_block = df_clean[self.feature_columns].copy()
        else:
            # При обучении используем все доступные фичи
            feature_block = df_clean[self.feature_columns].copy()
        
        feature_block = feature_block.ffill().bfill().fillna(0.0)
        feature_data = feature_block.values
        
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
        
        feature_block = df[self.feature_columns].copy()
        feature_block = feature_block.ffill().bfill().fillna(0.0)
        self.scaler.fit(feature_block.values)
        self.is_fitted = True
    
    def save_scaler(self, filepath: str, metadata: Optional[Dict] = None):
        """
        Сохраняет scaler и статистику для мониторинга аномалий
        
        Args:
            filepath: Путь для сохранения
            metadata: Дополнительные метаданные (настройки подготовки данных, версия и т.д.)
        """
        # Сохраняем mean и std для каждого фича (для мониторинга аномалий)
        feature_stats = {}
        if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
            for i, col in enumerate(self.feature_columns):
                feature_stats[col] = {
                    'mean': self.scaler.mean_[i],
                    'std': self.scaler.scale_[i]  # scale_ это std для StandardScaler
                }
        
        # Метаданные по умолчанию
        if metadata is None:
            metadata = {}
        
        # Добавляем информацию о фичах в метаданные
        metadata['num_features'] = len(self.feature_columns)
        metadata['feature_columns_hash'] = hash(tuple(sorted(self.feature_columns)))
        metadata['saved_at'] = pd.Timestamp.now().isoformat()
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length,
                'target_column': self.target_column,
                'feature_stats': feature_stats,  # Статистика для мониторинга аномалий
                'metadata': metadata  # Метаданные о подготовке данных
            }, f)
    
    def load_scaler(self, filepath: str, validate_features: bool = True, 
                   df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Загружает scaler с валидацией
        
        Args:
            filepath: Путь к файлу scaler
            validate_features: Проверять ли соответствие фичей
            df: DataFrame для валидации (если None, валидация пропускается)
        
        Returns:
            Словарь с метаданными из scaler файла
        
        Raises:
            ValueError: Если фичи не соответствуют сохраненным
        """
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.sequence_length = data['sequence_length']
        self.target_column = data['target_column']
        self.is_fitted = True
        
        metadata = data.get('metadata', {})
        
        # Валидация фичей
        if validate_features and df is not None:
            self._validate_features(df, self.feature_columns, metadata)
        
        return metadata
    
    def _validate_features(self, df: pd.DataFrame, saved_features: List[str], 
                          metadata: Dict):
        """
        Валидирует соответствие фичей в DataFrame сохраненным фичам
        
        Args:
            df: DataFrame для проверки
            saved_features: Список сохраненных фичей
            metadata: Метаданные из scaler
        
        Raises:
            ValueError: Если есть несоответствия
        """
        # Получаем фичи из DataFrame (исключая целевые переменные)
        exclude_patterns = [
            'target', 'label', 'direction', 'future_return', 
            'future_volatility', 'signal_class_name', 'max_future_return'
        ]
        df_features = [
            col for col in df.columns
            if col != self.target_column
            and not any(pattern in col.lower() for pattern in exclude_patterns)
            and col not in (self.exclude_columns or [])
        ]
        
        # Проверяем наличие всех сохраненных фичей
        missing_features = set(saved_features) - set(df_features)
        if missing_features:
            raise ValueError(
                f"❌ ОШИБКА: В DataFrame отсутствуют фичи, которые использовались при обучении:\n"
                f"   Отсутствуют: {sorted(list(missing_features))[:10]}{'...' if len(missing_features) > 10 else ''}\n"
                f"   Всего отсутствует: {len(missing_features)} из {len(saved_features)}\n"
                f"   Убедитесь, что данные подготовлены с теми же настройками, что и при обучении."
            )
        
        # Проверяем лишние фичи (предупреждение, не ошибка)
        extra_features = set(df_features) - set(saved_features)
        if extra_features:
            print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: В DataFrame есть фичи, которых не было при обучении:")
            print(f"   Лишние фичи: {sorted(list(extra_features))[:10]}{'...' if len(extra_features) > 10 else ''}")
            print(f"   Всего лишних: {len(extra_features)}")
            print(f"   Эти фичи будут проигнорированы при создании последовательностей.")
        
        # Проверяем порядок фичей (опционально, но важно для консистентности)
        if len(df_features) == len(saved_features):
            if list(df_features) != list(saved_features):
                print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Порядок фичей в DataFrame отличается от сохраненного.")
                print(f"   Порядок будет исправлен автоматически.")
        
        # Проверяем hash фичей из метаданных (если есть)
        if 'feature_columns_hash' in metadata:
            current_hash = hash(tuple(sorted(saved_features)))
            if current_hash != metadata['feature_columns_hash']:
                print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Hash фичей не совпадает с сохраненным.")
                print(f"   Возможно, файл scaler был изменен после сохранения.")
        
        print(f"✓ Валидация фичей пройдена: {len(saved_features)} фичей соответствуют сохраненным")

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
    
    # Создаем Dataset'ы с ленивой загрузкой для экономии памяти
    train_dataset = TimeSeriesDataset(train_sequences, train_targets, lazy_load=True)
    val_dataset = TimeSeriesDataset(val_sequences, val_targets, lazy_load=True)
    test_dataset = TimeSeriesDataset(test_sequences, test_targets, lazy_load=True)
    
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

