"""
Модуль для кэширования тиковых данных
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import os
import pickle
from pathlib import Path

class TickCache:
    """
    Класс для управления кэшем тиковых данных
    """
    
    def __init__(self, cache_dir: str = 'workspace/raw_data/ticks'):
        """
        Args:
            cache_dir: Директория для хранения кэша тиков
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'tick_metadata.pkl'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Загружает метаданные кэша"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Сохраняет метаданные кэша"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _get_tick_file_path(self, symbol: str, date: datetime, use_parquet: bool = True) -> Path:
        """Возвращает путь к файлу тиков для даты"""
        date_str = date.strftime('%Y%m%d')
        extension = '.parquet' if use_parquet else '.pkl'
        return self.cache_dir / f'{symbol}_{date_str}{extension}'
    
    def _get_date_range(self, start_date: datetime, end_date: datetime) -> list:
        """Возвращает список дат в диапазоне"""
        dates = []
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            dates.append(datetime.combine(current_date, datetime.min.time()))
            current_date += timedelta(days=1)
        
        return dates
    
    def get_cached_ticks(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Получает тики из кэша
        
        Args:
            symbol: Символ
            start_date: Начальная дата
            end_date: Конечная дата
        
        Returns:
            DataFrame с тиками или None если нет в кэше
        """
        dates = self._get_date_range(start_date, end_date)
        all_ticks = []
        
        for date in dates:
            # Сначала пытаемся загрузить parquet (новый формат)
            file_path_parquet = self._get_tick_file_path(symbol, date, use_parquet=True)
            file_path_pickle = self._get_tick_file_path(symbol, date, use_parquet=False)
            
            df = None
            file_path = None
            
            # Пробуем загрузить parquet (приоритет)
            if file_path_parquet.exists():
                try:
                    df = pd.read_parquet(file_path_parquet)
                    file_path = file_path_parquet
                except Exception as e:
                    # Если parquet не загрузился, пробуем pickle (обратная совместимость)
                    if file_path_pickle.exists():
                        try:
                            df = pd.read_pickle(file_path_pickle)
                            file_path = file_path_pickle
                            # Автоматически конвертируем в parquet для будущего использования
                            try:
                                df.to_parquet(file_path_parquet, compression='snappy', index=True)
                                # Удаляем старый pickle файл после успешной конвертации
                                file_path_pickle.unlink()
                            except Exception:
                                pass  # Если не удалось конвертировать, продолжаем с pickle
                        except (ModuleNotFoundError, ImportError, AttributeError) as e:
                            error_msg = str(e)
                            if 'numpy._core' in error_msg or 'numpy.core' in error_msg:
                                print(f"⚠️  Пропущен файл {file_path_pickle.name} (несовместимость версий numpy)")
                            else:
                                print(f"Ошибка при загрузке тиков из {file_path_pickle}: {e}")
                        except Exception as e:
                            print(f"Ошибка при загрузке тиков из {file_path_pickle}: {e}")
                    else:
                        print(f"Ошибка при загрузке parquet из {file_path_parquet}: {e}")
            # Если parquet нет, пробуем pickle (обратная совместимость)
            elif file_path_pickle.exists():
                try:
                    df = pd.read_pickle(file_path_pickle)
                    file_path = file_path_pickle
                    # Автоматически конвертируем в parquet для будущего использования
                    try:
                        df.to_parquet(file_path_parquet, compression='snappy', index=True)
                        # Удаляем старый pickle файл после успешной конвертации
                        file_path_pickle.unlink()
                    except Exception:
                        pass  # Если не удалось конвертировать, продолжаем с pickle
                except (ModuleNotFoundError, ImportError, AttributeError) as e:
                    error_msg = str(e)
                    if 'numpy._core' in error_msg or 'numpy.core' in error_msg:
                        print(f"⚠️  Пропущен файл {file_path_pickle.name} (несовместимость версий numpy)")
                    else:
                        print(f"Ошибка при загрузке тиков из {file_path_pickle}: {e}")
                except Exception as e:
                    print(f"Ошибка при загрузке тиков из {file_path_pickle}: {e}")
            
            # Если данные загружены, фильтруем по времени
            if df is not None and not df.empty:
                # Фильтруем по времени
                mask = (df.index >= start_date) & (df.index <= end_date)
                filtered_df = df[mask]
                if not filtered_df.empty:
                    all_ticks.append(filtered_df)
        
        if all_ticks:
            result = pd.concat(all_ticks).sort_index()
            return result
        
        return None
    
    def save_ticks(self, symbol: str, ticks_df: pd.DataFrame, batch_size: int = 10):
        """
        Сохраняет тики в кэш (по дням)
        
        Args:
            symbol: Символ
            ticks_df: DataFrame с тиками
            batch_size: Количество дней для батчевого сохранения
        """
        if ticks_df.empty:
            return
        
        # Получаем min/max даты БЕЗ копирования DataFrame
        min_date = ticks_df.index.min()
        max_date = ticks_df.index.max()
        
        # Группируем по дням БЕЗ полного копирования DataFrame
        # Используем groupby напрямую, обрабатывая группы по одной
        grouped = ticks_df.groupby(ticks_df.index.date)
        
        # Сохраняем по дням, обрабатывая батчами
        dates_list = []
        for date, group_df in grouped:
            dates_list.append(date)
            date_dt = datetime.combine(date, datetime.min.time())
            file_path = self._get_tick_file_path(symbol, date_dt, use_parquet=True)
            
            # Сохраняем группу в parquet формате (кроссплатформенный)
            try:
                group_df.to_parquet(file_path, compression='snappy', index=True)
                # Удаляем старый pickle файл, если существует
                old_pickle_path = self._get_tick_file_path(symbol, date_dt, use_parquet=False)
                if old_pickle_path.exists():
                    old_pickle_path.unlink()
            except Exception as e:
                print(f"    Ошибка при сохранении {file_path}: {e}")
        
        # Обновляем метаданные один раз в конце
        if symbol not in self.metadata:
            self.metadata[symbol] = {}
        
        if 'min_date' not in self.metadata[symbol] or min_date < self.metadata[symbol]['min_date']:
            self.metadata[symbol]['min_date'] = min_date
        
        if 'max_date' not in self.metadata[symbol] or max_date > self.metadata[symbol]['max_date']:
            self.metadata[symbol]['max_date'] = max_date
        
        self.metadata[symbol]['last_update'] = datetime.now()
        self._save_metadata()
    
    def get_cache_coverage(self, symbol: str) -> Optional[Tuple[datetime, datetime]]:
        """
        Возвращает диапазон дат, покрытый кэшем
        
        Args:
            symbol: Символ
        
        Returns:
            Tuple (min_date, max_date) или None
        """
        if symbol not in self.metadata:
            return None
        
        meta = self.metadata[symbol]
        if 'min_date' in meta and 'max_date' in meta:
            return meta['min_date'], meta['max_date']
        
        return None
    
    def needs_update(self, symbol: str, required_start: datetime, 
                    required_end: datetime) -> Tuple[bool, Optional[datetime], Optional[datetime]]:
        """
        Проверяет, нужно ли обновить кэш
        
        Args:
            symbol: Символ
            required_start: Требуемая начальная дата
            required_end: Требуемая конечная дата
        
        Returns:
            Tuple (needs_update, missing_start, missing_end)
        """
        coverage = self.get_cache_coverage(symbol)
        
        if coverage is None:
            # Кэша нет вообще
            return True, required_start, required_end
        
        cache_min, cache_max = coverage
        
        # Проверяем, покрывает ли кэш требуемый диапазон
        needs_start = required_start < cache_min
        needs_end = required_end > cache_max
        
        if needs_start or needs_end:
            missing_start = required_start if needs_start else cache_min
            missing_end = required_end if needs_end else cache_max
            return True, missing_start, missing_end
        
        return False, None, None
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Очищает кэш
        
        Args:
            symbol: Символ для очистки (если None - очищает весь кэш)
        """
        if symbol is None:
            # Очищаем весь кэш (и pickle, и parquet файлы)
            for file in self.cache_dir.glob('*.pkl'):
                if file.name != 'tick_metadata.pkl':
                    file.unlink()
            for file in self.cache_dir.glob('*.parquet'):
                file.unlink()
            self.metadata = {}
        else:
            # Очищаем кэш для конкретного символа (и pickle, и parquet файлы)
            for file in self.cache_dir.glob(f'{symbol}_*.pkl'):
                file.unlink()
            for file in self.cache_dir.glob(f'{symbol}_*.parquet'):
                file.unlink()
            if symbol in self.metadata:
                del self.metadata[symbol]
        
        self._save_metadata()
    
    def get_cache_size(self, symbol: Optional[str] = None) -> int:
        """
        Возвращает размер кэша в файлах
        
        Args:
            symbol: Символ (если None - для всех)
        
        Returns:
            Количество файлов
        """
        if symbol is None:
            # Подсчитываем и pickle, и parquet файлы
            pkl_files = [f for f in self.cache_dir.glob('*.pkl') 
                        if f.name != 'tick_metadata.pkl']
            parquet_files = list(self.cache_dir.glob('*.parquet'))
            return len(pkl_files) + len(parquet_files)
        else:
            # Подсчитываем и pickle, и parquet файлы для символа
            pkl_files = list(self.cache_dir.glob(f'{symbol}_*.pkl'))
            parquet_files = list(self.cache_dir.glob(f'{symbol}_*.parquet'))
            return len(pkl_files) + len(parquet_files)

