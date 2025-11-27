"""
Модуль для генерации целевых переменных для классификации
Пробой, Отскок, Неопределенность
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple

class TargetGenerator:
    """
    Генератор целевых переменных для классификации торговых сигналов
    """
    
    def __init__(self, 
                 breakout_threshold: float = 450.0,
                 bounce_threshold: float = 350.0,
                 lookahead_periods: int = 60):
        """
        Args:
            breakout_threshold: Порог для пробоя в пунктах (по умолчанию 200)
            bounce_threshold: Порог для отскока в пунктах (по умолчанию 150)
            lookahead_periods: Количество периодов вперед для анализа (по умолчанию 60 минут)
        """
        self.breakout_threshold = breakout_threshold
        self.bounce_threshold = bounce_threshold
        self.lookahead_periods = lookahead_periods
    
    def calculate_future_returns(self, df: pd.DataFrame, 
                                 price_column: str = 'close') -> pd.DataFrame:
        """
        Вычисляет будущие доходности для разных периодов
        
        Args:
            df: DataFrame с ценовыми данными
            price_column: Название колонки с ценой
        
        Returns:
            DataFrame с добавленными колонками future_return_N
        """
        df = df.copy()
        current_price = df[price_column]
        
        # Вычисляем доходности для разных периодов вперед
        for period in [1, 5, 10, 20, 30, 60]:
            future_price = df[price_column].shift(-period)
            # Доходность в пунктах (для золота обычно 0.01 = 1 пункт)
            df[f'future_return_{period}'] = (future_price - current_price) * 100
        
        return df
    
    def classify_signal(self, df: pd.DataFrame, 
                       price_column: str = 'close') -> pd.Series:
        """
        Классифицирует сигналы: пробой вверх/вниз, отскок вверх/вниз, неопределенность
        
        Args:
            df: DataFrame с ценовыми данными и future_return колонками
            price_column: Название колонки с ценой
        
        Returns:
            Series с метками классов:
            0 = Неопределенность
            1 = Пробой вверх (BUY)
            2 = Пробой вниз (SELL)
            3 = Отскок вверх (BUY после падения)
            4 = Отскок вниз (SELL после роста)
        """
        signals = pd.Series(index=df.index, dtype=int)
        
        # Получаем максимальную доходность в будущем
        future_returns = []
        for period in [1, 5, 10, 20, 30, 60]:
            if f'future_return_{period}' in df.columns:
                future_returns.append(df[f'future_return_{period}'].abs())
        
        if not future_returns:
            return signals.fillna(0)
        
        future_returns_df = pd.concat(future_returns, axis=1)
        # Заполняем NaN перед вычислением max и idxmax, чтобы избежать FutureWarning
        future_returns_df_filled = future_returns_df.fillna(-np.inf)
        max_return = future_returns_df_filled.max(axis=1)
        max_return_period = future_returns_df_filled.idxmax(axis=1)
        
        # Анализируем каждый временной шаг
        for i in range(len(df)):
            if pd.isna(max_return.iloc[i]):
                signals.iloc[i] = 0  # Неопределенность
                continue
            
            max_ret = max_return.iloc[i]
            period_idx = max_return_period.iloc[i]
            
            # Получаем доходность на этом периоде
            period_num = int(period_idx.split('_')[-1])
            period_return = df[f'future_return_{period_num}'].iloc[i]
            
            # СНАЧАЛА проверяем на отскок (приоритет)
            bounce_result = self._check_bounce(df, i, period_num, period_return)
            if bounce_result is not None:
                # bounce_result может быть 'up' или 'down'
                if bounce_result == 'up':
                    signals.iloc[i] = 3  # Отскок вверх
                elif bounce_result == 'down':
                    signals.iloc[i] = 4  # Отскок вниз
                continue
            
            # Затем проверяем на пробой
            # Пробой должен быть достаточно сильным и устойчивым
            if abs(period_return) >= self.breakout_threshold:
                is_breakout = self._check_breakout(df, i, period_num, period_return)
                if is_breakout:
                    if period_return > 0:
                        signals.iloc[i] = 1  # Пробой вверх
                    else:
                        signals.iloc[i] = 2  # Пробой вниз
                    continue
            
            # Иначе - неопределенность
            signals.iloc[i] = 0
        
        return signals
    
    def _check_breakout(self, df: pd.DataFrame, idx: int, 
                       period: int, period_return: float) -> bool:
        """
        Проверяет, является ли движение пробоем
        
        Пробой: движение > threshold в одном направлении без значительного разворота
        """
        if idx + period >= len(df):
            return False
        
        # Проверяем промежуточные доходности
        direction = np.sign(period_return)
        
        # Проверяем, что движение было в основном в одном направлении
        intermediate_returns = []
        for p in [1, 5, 10, 20]:
            if p < period and idx + p < len(df):
                if f'future_return_{p}' in df.columns:
                    ret = df[f'future_return_{p}'].iloc[idx]
                    intermediate_returns.append(ret)
        
        if intermediate_returns:
            # Проверяем, что промежуточные движения не противоречат основному
            same_direction = sum(1 for r in intermediate_returns 
                                if np.sign(r) == direction) / len(intermediate_returns)
            
            # Если более 80% движений в одном направлении - это пробой
            # Ужесточили с 70% до 80% для более строгой фильтрации
            if same_direction >= 0.8:
                return True
        
        # Если нет промежуточных данных, проверяем только финальную доходность
        return abs(period_return) >= self.breakout_threshold
    
    def _check_bounce(self, df: pd.DataFrame, idx: int, 
                     period: int, period_return: float) -> Optional[str]:
        """
        Проверяет, является ли движение отскоком
        
        Отскок: движение в одну сторону, затем разворот > bounce_threshold
        
        Returns:
            'up' если отскок вверх (после падения), 
            'down' если отскок вниз (после роста),
            None если это не отскок
        """
        if idx + period >= len(df):
            return None
        
        # Проверяем промежуточные доходности до 70% периода
        max_check_period = int(period * 0.7)
        intermediate_returns = []
        intermediate_periods = []
        
        for p in [1, 5, 10, 20, 30]:
            if p <= max_check_period and idx + p < len(df):
                if f'future_return_{p}' in df.columns:
                    ret = df[f'future_return_{p}'].iloc[idx]
                    intermediate_returns.append(ret)
                    intermediate_periods.append(p)
        
        if len(intermediate_returns) < 2:
            return None
        
        # Ищем максимальное движение в начальном направлении (до 70% периода)
        max_initial_movement = max([abs(r) for r in intermediate_returns])
        max_initial_idx = np.argmax([abs(r) for r in intermediate_returns])
        initial_direction = np.sign(intermediate_returns[max_initial_idx])
        
        # Проверяем, было ли начальное движение достаточно сильным
        # Минимальное движение должно быть не менее 30% от порога отскока
        min_initial_movement = self.bounce_threshold * 0.3
        if max_initial_movement < min_initial_movement:
            return None
        
        # Проверяем финальную доходность (должна быть противоположного знака)
        final_direction = np.sign(period_return)
        
        # Отскок: начальное движение в одну сторону, затем разворот
        if initial_direction != 0 and final_direction != 0:
            if initial_direction != final_direction:
                # Проверяем, что разворот достаточно сильный
                if abs(period_return) >= self.bounce_threshold:
                    # Определяем направление отскока
                    if final_direction > 0:  # Разворот вверх после падения
                        return 'up'
                    else:  # Разворот вниз после роста
                        return 'down'
        
        return None
    
    def generate_targets(self, df: pd.DataFrame, 
                       price_column: str = 'close') -> pd.DataFrame:
        """
        Генерирует целевые переменные для всего DataFrame
        
        Args:
            df: DataFrame с ценовыми данными
            price_column: Название колонки с ценой
        
        Returns:
            DataFrame с добавленными целевыми переменными:
            - future_return_N: доходности на разных периодах
            - signal_class: класс сигнала (0=неопределенность, 1=пробой, 2=отскок)
            - signal_class_name: название класса
        """
        df = df.copy()
        
        # Вычисляем будущие доходности
        df = self.calculate_future_returns(df, price_column)
        
        # Классифицируем сигналы
        signals = self.classify_signal(df, price_column)
        df['signal_class'] = signals
        
        # Добавляем названия классов
        class_names = {
            0: 'uncertainty',
            1: 'breakout_up',
            2: 'breakout_down',
            3: 'bounce_up',
            4: 'bounce_down'
        }
        df['signal_class_name'] = df['signal_class'].map(class_names)
        
        # Добавляем метрики для анализа
        df['max_future_return'] = df[[col for col in df.columns 
                                      if col.startswith('future_return_')]].abs().max(axis=1)
        
        return df
    
    def get_class_distribution(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает распределение классов
        
        Args:
            df: DataFrame с колонкой signal_class
        
        Returns:
            Series с количеством каждого класса
        """
        if 'signal_class' not in df.columns:
            raise ValueError("DataFrame должен содержать колонку 'signal_class'")
        
        return df['signal_class'].value_counts().sort_index()
    
    def balance_classes(self, df: pd.DataFrame, 
                      method: str = 'undersample') -> pd.DataFrame:
        """
        Балансирует классы (если нужно)
        
        Args:
            df: DataFrame с колонкой signal_class
            method: Метод балансировки ('undersample' или 'oversample')
        
        Returns:
            Сбалансированный DataFrame
        """
        if 'signal_class' not in df.columns:
            raise ValueError("DataFrame должен содержать колонку 'signal_class'")
        
        class_counts = df['signal_class'].value_counts()
        min_count = class_counts.min()
        
        if method == 'undersample':
            # Undersampling: берем одинаковое количество из каждого класса
            balanced_dfs = []
            for class_val in class_counts.index:
                class_df = df[df['signal_class'] == class_val]
                balanced_dfs.append(class_df.sample(n=min_count, random_state=42))
            
            return pd.concat(balanced_dfs).sort_index()
        
        elif method == 'oversample':
            # Oversampling: дублируем меньшие классы
            balanced_dfs = []
            for class_val in class_counts.index:
                class_df = df[df['signal_class'] == class_val]
                if len(class_df) < min_count:
                    # Дублируем с добавлением шума
                    n_samples = min_count - len(class_df)
                    additional = class_df.sample(n=n_samples, replace=True, random_state=42)
                    balanced_dfs.append(pd.concat([class_df, additional]))
                else:
                    balanced_dfs.append(class_df.sample(n=min_count, random_state=42))
            
            return pd.concat(balanced_dfs).sort_index()
        
        else:
            raise ValueError(f"Неизвестный метод балансировки: {method}")

