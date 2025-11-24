"""
Модуль для нормализации и масштабирования данных
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
import os

class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Класс для масштабирования фичей с сохранением параметров
    """
    
    def __init__(self, method: str = 'standard', feature_columns: Optional[List[str]] = None):
        """
        Args:
            method: Метод масштабирования ('standard', 'minmax', 'robust')
            feature_columns: Список колонок для масштабирования (если None, все числовые)
        """
        self.method = method
        self.feature_columns = feature_columns
        self.scaler = None
        self.fitted_columns = None
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Неизвестный метод масштабирования: {method}")
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Обучение scaler на данных
        
        Args:
            X: DataFrame с фичами
            y: Не используется (для совместимости с sklearn)
        """
        if self.feature_columns is None:
            # Выбираем только числовые колонки
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            # Исключаем целевые переменные и временные фичи
            exclude_patterns = ['target', 'label', 'direction', 'future_return']
            self.fitted_columns = [
                col for col in numeric_cols 
                if not any(pattern in col.lower() for pattern in exclude_patterns)
            ]
        else:
            self.fitted_columns = [col for col in self.feature_columns if col in X.columns]
        
        if len(self.fitted_columns) == 0:
            raise ValueError("Не найдено колонок для масштабирования")
        
        self.scaler.fit(X[self.fitted_columns])
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Применение масштабирования
        
        Args:
            X: DataFrame с фичами
        
        Returns:
            Масштабированный DataFrame
        """
        X = X.copy()
        
        if self.fitted_columns is None:
            raise ValueError("Scaler не был обучен. Вызовите fit() сначала.")
        
        # Масштабируем только указанные колонки
        X[self.fitted_columns] = self.scaler.transform(X[self.fitted_columns])
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Обучение и применение масштабирования"""
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Обратное преобразование
        
        Args:
            X: Масштабированный DataFrame
        
        Returns:
            DataFrame с исходными значениями
        """
        X = X.copy()
        
        if self.fitted_columns is None:
            raise ValueError("Scaler не был обучен.")
        
        X[self.fitted_columns] = self.scaler.inverse_transform(X[self.fitted_columns])
        return X
    
    def save(self, filepath: str):
        """Сохранение scaler в файл"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'fitted_columns': self.fitted_columns,
                'method': self.method
            }, f)
    
    def load(self, filepath: str):
        """Загрузка scaler из файла"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.fitted_columns = data['fitted_columns']
            self.method = data['method']
        return self

def normalize_features(df: pd.DataFrame, method: str = 'standard',
                      exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Нормализация фичей в DataFrame
    
    Args:
        df: DataFrame с фичами
        method: Метод нормализации ('standard', 'minmax', 'robust')
        exclude_columns: Колонки для исключения из нормализации
    
    Returns:
        Нормализованный DataFrame
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Выбираем числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Исключаем указанные колонки
    cols_to_scale = [col for col in numeric_cols if col not in exclude_columns]
    
    if len(cols_to_scale) == 0:
        return df
    
    df_scaled = df.copy()
    
    if method == 'standard':
        scaler = StandardScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    elif method == 'robust':
        scaler = RobustScaler()
        df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        raise ValueError(f"Неизвестный метод нормализации: {method}")
    
    return df_scaled

def remove_outliers(df: pd.DataFrame, columns: Optional[List[str]] = None,
                   method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
    """
    Удаление выбросов из данных
    
    Args:
        df: DataFrame
        columns: Колонки для обработки (если None, все числовые)
        method: Метод ('iqr' или 'zscore')
        threshold: Порог для метода zscore
    
    Returns:
        DataFrame с удаленными выбросами (заменены на NaN)
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-10))
            df.loc[z_scores > threshold, col] = np.nan
    
    return df

def fill_missing_values(df: pd.DataFrame, method: str = 'forward',
                       limit: Optional[int] = None) -> pd.DataFrame:
    """
    Заполнение пропущенных значений
    
    Args:
        df: DataFrame
        method: Метод заполнения ('forward', 'backward', 'mean', 'median', 'zero')
        limit: Максимальное количество последовательных пропусков для заполнения
    
    Returns:
        DataFrame с заполненными значениями
    """
    df = df.copy()
    
    if method == 'forward':
        df = df.fillna(method='ffill', limit=limit)
    elif method == 'backward':
        df = df.fillna(method='bfill', limit=limit)
    elif method == 'mean':
        df = df.fillna(df.mean())
    elif method == 'median':
        df = df.fillna(df.median())
    elif method == 'zero':
        df = df.fillna(0)
    else:
        raise ValueError(f"Неизвестный метод заполнения: {method}")
    
    return df

