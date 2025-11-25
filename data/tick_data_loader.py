"""
Модуль для загрузки тиковых данных из MetaTrader 5
"""
import pandas as pd
import numpy as np
import time
import pickle
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import MetaTrader5 as mt5
from data.tick_cache import TickCache

class TickDataLoader:
    """Класс для загрузки тиковых данных из MetaTrader 5"""
    
    @staticmethod
    def _get_timestamp() -> str:
        """Возвращает форматированную временную метку"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Форматирует длительность в читаемый вид"""
        if seconds < 60:
            return f"{seconds:.1f}с"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}м {secs}с"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}ч {minutes}м {secs}с"
    
    def __init__(self, mt5_connection=None, use_cache: bool = True, cache_dir: str = 'data/ticks'):
        """
        Инициализация загрузчика тиков
        
        Args:
            mt5_connection: Существующее подключение MT5 (опционально)
            use_cache: Использовать кэш тиков
            cache_dir: Директория для кэша
        """
        self.mt5_connection = mt5_connection
        self.connected = False
        self.use_cache = use_cache
        self.cache = TickCache(cache_dir) if use_cache else None
        self.cache_dir = Path(cache_dir)
        self.default_lookback_days = 540  # 1.5 года в днях
    
    def _ensure_connected(self):
        """Убеждается, что MT5 подключен"""
        if not self.connected:
            # Если передан существующий объект подключения, проверяем его
            if self.mt5_connection is not None:
                # Если это объект MT5DataLoader, проверяем его подключение
                if hasattr(self.mt5_connection, 'connected') and self.mt5_connection.connected:
                    self.connected = True
                    return
            # Иначе инициализируем новое подключение
            if not mt5.initialize():
                raise ConnectionError("Не удалось инициализировать MT5")
            self.connected = True
    
    def load_ticks(self, symbol: str, start_time: datetime, 
                   end_time: datetime, flags: int = mt5.COPY_TICKS_ALL,
                   use_cache: Optional[bool] = None) -> pd.DataFrame:
        """
        Загрузка тиковых данных за указанный период
        
        Args:
            symbol: Символ валютной пары (например, 'EURUSD')
            start_time: Начальное время
            end_time: Конечное время
            flags: Флаги загрузки (COPY_TICKS_ALL, COPY_TICKS_INFO, COPY_TICKS_TRADE)
            use_cache: Использовать кэш (если None, используется self.use_cache)
        
        Returns:
            DataFrame с колонками: time, bid, ask, volume, flags
        """
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Пытаемся загрузить из кэша
        if use_cache and self.cache is not None:
            cached_ticks = self.cache.get_cached_ticks(symbol, start_time, end_time)
            if cached_ticks is not None and not cached_ticks.empty:
                return cached_ticks
        
        # Загружаем из MT5
        self._ensure_connected()
        
        # Убеждаемся, что символ доступен
        if not mt5.symbol_select(symbol, True):
            print(f"Предупреждение: символ {symbol} не найден или не активирован")
            return pd.DataFrame()
        
        ticks = mt5.copy_ticks_range(symbol, start_time, end_time, flags)
        
        if ticks is None or len(ticks) == 0:
            return pd.DataFrame()
        
        # Преобразование в DataFrame
        df = pd.DataFrame(ticks)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Расчет спреда
        df['spread'] = df['ask'] - df['bid']
        df['spread_pips'] = df['spread'] * 10000  # Для большинства валютных пар
        
        # Сохраняем в кэш
        if use_cache and self.cache is not None:
            self.cache.save_ticks(symbol, df)
        
        return df
    
    def load_ticks_for_minute(self, symbol: str, minute_time: datetime,
                              lookback_minutes: int = 1) -> pd.DataFrame:
        """
        Загрузка тиков за последние N минут перед указанной минутой
        
        Args:
            symbol: Символ валютной пары
            minute_time: Время минутной свечи
            lookback_minutes: Количество минут для загрузки тиков
        
        Returns:
            DataFrame с тиковыми данными
        """
        end_time = minute_time
        start_time = minute_time - timedelta(minutes=lookback_minutes)
        
        # Используем кэш при загрузке
        return self.load_ticks(symbol, start_time, end_time, use_cache=self.use_cache)
    
    def ensure_cache_coverage(self, symbol: str, required_end: datetime):
        """
        Обеспечивает покрытие кэша до указанной даты (минимум 1.5 года назад)
        
        Args:
            symbol: Символ
            required_end: Требуемая конечная дата
        """
        if not self.use_cache or self.cache is None:
            return
        
        # Вычисляем требуемую начальную дату (1.5 года назад от required_end)
        required_start = required_end - timedelta(days=self.default_lookback_days)
        
        # Проверяем, нужно ли обновить кэш
        needs_update, missing_start, missing_end = self.cache.needs_update(
            symbol, required_start, required_end
        )
        
        if not needs_update:
            timestamp = self._get_timestamp()
            print(f"  [{timestamp}] Кэш тиков для {symbol} актуален")
            return
        
        timestamp = self._get_timestamp()
        print(f"  [{timestamp}] Обновление кэша тиков для {symbol}...")
        print(f"    [{timestamp}] Требуемый диапазон: {missing_start} - {missing_end}")
        
        # Загружаем недостающие тики
        self._ensure_connected()
        
        if not mt5.symbol_select(symbol, True):
            print(f"  Предупреждение: символ {symbol} не найден")
            return
        
        # Загружаем тики порциями по дням для надежности
        current_date = missing_start
        loaded_days = 0
        total_ticks = 0
        errors = []
        
        days_to_load = (missing_end - missing_start).days + 1
        timestamp = self._get_timestamp()
        print(f"    [{timestamp}] Начало загрузки: {current_date}")
        print(f"    [{timestamp}] Конец загрузки: {missing_end}")
        print(f"    [{timestamp}] Всего дней для загрузки: {days_to_load}")
        
        # Для больших диапазонов ограничиваем загрузку последними 30 днями (чтобы не ждать долго)
        # В продакшене можно убрать это ограничение
        if days_to_load > 550:
            print(f"    ⚠ Большой диапазон. Загружаем только последние 30 дней для теста")
            print(f"    Для полной загрузки уберите ограничение в коде")
            current_date = max(missing_start, missing_end - timedelta(days=30))
        
        # Собираем все тики в один DataFrame для батчевого сохранения
        all_ticks_list = []
        start_time = time.time()
        
        timestamp = self._get_timestamp()
        print(f"    [{timestamp}] Нажмите Ctrl+C для прерывания (уже загруженные данные сохранятся)")
        
        # Информационное сообщение о пропусках
        if days_to_load > 10:
            print(f"    [{timestamp}] ℹ Примечание: дни без тиков (выходные, старые даты, когда рынок был закрыт)")
            print(f"    [{timestamp}]    будут показываться как 'загружено с тиками: 0' - это нормально.")
            print(f"    [{timestamp}]    Обработка продолжается, тики будут загружены для доступных торговых дней.")
        
        # Переменная для отслеживания последнего вывода прогресса
        last_progress_day = -1
        
        try:
            while current_date <= missing_end:
                day_end = min(current_date + timedelta(days=1), missing_end)
                prev_date = current_date  # Сохраняем для проверки
                
                try:
                    ticks = None
                    
                    # Метод 1: copy_ticks_range
                    try:
                        ticks = mt5.copy_ticks_range(symbol, current_date, day_end, mt5.COPY_TICKS_ALL)
                    except Exception as e1:
                        pass
                    
                    # Метод 2: copy_ticks_from (для более свежих данных)
                    if (ticks is None or len(ticks) == 0) and current_date >= datetime.now() - timedelta(days=7):
                        try:
                            seconds = int((day_end - current_date).total_seconds())
                            if seconds > 0:
                                ticks = mt5.copy_ticks_from(symbol, current_date, seconds, mt5.COPY_TICKS_ALL)
                        except Exception as e2:
                            pass
                    
                    if ticks is not None and len(ticks) > 0:
                        df = pd.DataFrame(ticks)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                        
                        # Фильтруем по нужному диапазону
                        mask = (df.index >= current_date) & (df.index <= day_end)
                        df = df[mask]
                        
                        if not df.empty:
                            df['spread'] = df['ask'] - df['bid']
                            df['spread_pips'] = df['spread'] * 10000
                            
                            # Добавляем в список для батчевого сохранения
                            all_ticks_list.append(df)
                            loaded_days += 1
                            total_ticks += len(df)
                
                except Exception as e:
                    errors.append(str(e))
                    if len(errors) <= 3:
                        print(f"    Ошибка при загрузке за {current_date.date()}: {e}")
                
                # Обновляем счетчик обработанных дней (даже если тиков нет)
                processed_days = int((current_date - missing_start).days) + 1
                
                # Показываем прогресс только если обработали новый день (не повторяем)
                if processed_days != last_progress_day and (processed_days % 3 == 0 or processed_days == days_to_load):
                    elapsed = time.time() - start_time
                    timestamp = self._get_timestamp()
                    duration_str = self._format_duration(elapsed)
                    skipped_days = processed_days - loaded_days
                    if skipped_days > 0:
                        print(f"    [{timestamp}] Обработано дней: {processed_days}/{days_to_load}, загружено с тиками: {loaded_days} (пропущено: {skipped_days}), тиков: {total_ticks:,}, время работы: {duration_str}")
                    else:
                        print(f"    [{timestamp}] Обработано дней: {processed_days}/{days_to_load}, загружено с тиками: {loaded_days}, тиков: {total_ticks:,}, время работы: {duration_str}")
                    last_progress_day = processed_days
                
                # Сохраняем батчами каждые 3 дня, чтобы не накапливать слишком много в памяти
                if len(all_ticks_list) >= 3:
                    try:
                        save_start = time.time()
                        combined_df = pd.concat(all_ticks_list)
                        self.cache.save_ticks(symbol, combined_df, batch_size=3)
                        all_ticks_list = []  # Очищаем список
                        save_time = time.time() - save_start
                        timestamp = self._get_timestamp()
                        print(f"    [{timestamp}] ✓ Сохранено в кэш ({loaded_days} дней, {save_time:.1f}с)")
                    except Exception as e:
                        print(f"    ⚠ Ошибка при сохранении батча: {e}")
                        all_ticks_list = []  # Очищаем даже при ошибке
                
                # Всегда обновляем current_date, даже если тиков нет или была ошибка
                current_date = day_end
                
                # Защита от бесконечного цикла: если дата не изменилась, принудительно увеличиваем
                if current_date == prev_date:
                    current_date = current_date + timedelta(days=1)
                    print(f"    ⚠ Принудительное увеличение даты (защита от зацикливания)")
                
                # Проверка на выход из цикла
                if current_date > missing_end:
                    break
                
                # Добавляем небольшую задержку для выходных дней (когда тиков нет)
                if ticks is None or len(ticks) == 0:
                    time.sleep(0.01)  # Небольшая задержка для выходных
            
            # Сохраняем оставшиеся тики
            if all_ticks_list:
                combined_df = pd.concat(all_ticks_list)
                self.cache.save_ticks(symbol, combined_df, batch_size=3)
                timestamp = self._get_timestamp()
                print(f"    [{timestamp}] Финальное сохранение в кэш...")
        
        except KeyboardInterrupt:
            print(f"\n    ⚠ Прерывание загрузки пользователем...")
            if all_ticks_list:
                print(f"    Сохранение уже загруженных данных...")
                combined_df = pd.concat(all_ticks_list)
                self.cache.save_ticks(symbol, combined_df, batch_size=3)
                print(f"    ✓ Сохранено {loaded_days} дней, тиков: {total_ticks:,}")
            raise
        
        if loaded_days > 0:
            timestamp = self._get_timestamp()
            total_elapsed = time.time() - start_time if 'start_time' in locals() else 0
            duration_str = self._format_duration(total_elapsed)
            print(f"  [{timestamp}] ✓ Кэш обновлен: загружено {loaded_days} дней, всего тиков: {total_ticks:,}, время: {duration_str}")
        else:
            print(f"  ⚠ Тики не загружены. Возможные причины:")
            print(f"    - Тиковая история недоступна в MT5 для этого периода")
            print(f"    - Символ {symbol} не активирован или недоступен")
            print(f"    - Тиковая история не включена в настройках MT5")
            print(f"    - Период слишком старый (тики доступны обычно только за последние месяцы)")
            if errors:
                print(f"    - Примеры ошибок: {errors[0]}")
            print(f"    Продолжаем без тиковых данных...")
        
        timestamp = self._get_timestamp()
        print(f"  [{timestamp}] Кэш обновлен: загружено {loaded_days} дней")
    
    def _get_progress_file_path(self, symbol: str) -> Path:
        """Возвращает путь к файлу прогресса обработки свечей"""
        return self.cache_dir / f'{symbol}_batch_progress.pkl'
    
    def _get_progress_dir(self, symbol: str) -> Path:
        """Директория для временных чанков прогресса"""
        return self.cache_dir / f'{symbol}_progress'
    
    def _load_progress_metadata(self, progress_file: Path) -> Dict:
        """Загружает метаданные прогресса без самих датафреймов"""
        if progress_file.exists():
            try:
                with open(progress_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                timestamp = self._get_timestamp()
                print(f"    [{timestamp}] ⚠ Не удалось прочитать метаданные прогресса: {e}")
        return {}
    
    def _save_batch_progress(self, symbol: str, ticks_data: Dict[datetime, pd.DataFrame], 
                            processed_times: list, dirty_minutes: Optional[List[datetime]] = None):
        """Сохраняет прогресс обработки свечей"""
        if not self.use_cache:
            return
        
        dirty_minutes = dirty_minutes or []
        
        progress_file = self._get_progress_file_path(symbol)
        progress_dir = self._get_progress_dir(symbol)
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = self._load_progress_metadata(progress_file)
        chunk_counter = metadata.get('chunk_counter', 0)
        chunks = metadata.get('chunks', [])
        
        if dirty_minutes:
            chunk_counter += 1
            chunk_filename = f'chunk_{chunk_counter:05d}.pkl'
            chunk_path = progress_dir / chunk_filename
            
            chunk_data = {}
            for minute in dirty_minutes:
                if minute in ticks_data:
                    chunk_data[minute] = ticks_data[minute]
            
            try:
                with open(chunk_path, 'wb') as chunk_file:
                    pickle.dump(chunk_data, chunk_file, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                timestamp = self._get_timestamp()
                print(f"    [{timestamp}] ⚠ Ошибка при сохранении чанка прогресса: {e}")
            else:
                chunks.append({
                    'filename': chunk_filename,
                    'minutes': dirty_minutes.copy(),
                    'saved_at': datetime.now()
                })
        
        progress_data = {
            'processed_times': processed_times,
            'last_update': datetime.now(),
            'symbol': symbol,
            'chunk_counter': chunk_counter,
            'chunks': chunks
        }
        
        try:
            with open(progress_file, 'wb') as f:
                pickle.dump(progress_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"    ⚠ Ошибка при сохранении прогресса: {e}")
    
    def _load_batch_progress(self, symbol: str) -> Optional[Dict]:
        """Загружает сохраненный прогресс обработки свечей"""
        if not self.use_cache:
            return None
        
        progress_file = self._get_progress_file_path(symbol)
        if not progress_file.exists():
            return None
        
        try:
            with open(progress_file, 'rb') as f:
                progress_data = pickle.load(f)
                # Проверяем, что это прогресс для правильного символа
                if progress_data.get('symbol') == symbol:
                    ticks_data = {}
                    missing_chunks = []
                    progress_dir = self._get_progress_dir(symbol)
                    for chunk_info in progress_data.get('chunks', []):
                        chunk_path = progress_dir / chunk_info.get('filename', '')
                        if chunk_path.exists():
                            try:
                                with open(chunk_path, 'rb') as chunk_file:
                                    chunk_data = pickle.load(chunk_file)
                                    ticks_data.update(chunk_data)
                            except Exception as chunk_error:
                                timestamp = self._get_timestamp()
                                print(f"    [{timestamp}] ⚠ Ошибка при чтении чанка {chunk_path}: {chunk_error}")
                        else:
                            missing_chunks.append(chunk_info.get('filename'))
                    
                    if missing_chunks:
                        timestamp = self._get_timestamp()
                        print(f"    [{timestamp}] ⚠ Отсутствуют файлы чанков прогресса: {', '.join(missing_chunks)}")
                    
                    progress_data['ticks_data'] = ticks_data
                    return progress_data
        except Exception as e:
            print(f"    ⚠ Ошибка при загрузке прогресса: {e}")
        
        return None
    
    def _clear_batch_progress(self, symbol: str):
        """Удаляет файл прогресса (после успешного завершения)"""
        if not self.use_cache:
            return
        
        progress_file = self._get_progress_file_path(symbol)
        progress_dir = self._get_progress_dir(symbol)
        if progress_file.exists():
            try:
                progress_file.unlink()
            except Exception as e:
                print(f"    ⚠ Ошибка при удалении файла прогресса: {e}")
        
        if progress_dir.exists():
            try:
                shutil.rmtree(progress_dir)
            except Exception as e:
                print(f"    ⚠ Ошибка при очистке директории прогресса: {e}")
    
    def load_ticks_batch(self, symbol: str, minute_times: pd.DatetimeIndex,
                        lookback_minutes: int = 1, 
                        save_progress_every: int = 1000,
                        resume: bool = True) -> Dict[datetime, pd.DataFrame]:
        """
        Загрузка тиков для множества минутных свечей с сохранением прогресса
        
        Args:
            symbol: Символ валютной пары
            minute_times: DatetimeIndex с временами минутных свечей
            lookback_minutes: Количество минут для загрузки тиков
            save_progress_every: Сохранять прогресс каждые N свечей
            resume: Продолжить с сохраненного прогресса, если он есть
        
        Returns:
            Словарь {minute_time: DataFrame с тиками}
        """
        # Обеспечиваем покрытие кэша
        if self.use_cache and self.cache is not None and len(minute_times) > 0:
            max_time = minute_times.max()
            self.ensure_cache_coverage(symbol, max_time)
        
        # Засекаем время начала обработки
        start_time = time.time()
        start_timestamp = self._get_timestamp()
        
        # Пытаемся загрузить сохраненный прогресс
        ticks_data = {}
        processed_times = set()
        dirty_minutes = []
        start_index = 0
        
        if resume:
            progress = self._load_batch_progress(symbol)
            if progress is not None:
                saved_ticks = progress.get('ticks_data', {})
                saved_times = set(progress.get('processed_times', []))
                
                if saved_ticks and saved_times:
                    ticks_data = saved_ticks
                    processed_times = saved_times
                    
                    # Находим индекс, с которого нужно продолжить
                    minute_times_list = list(minute_times)
                    for i, minute_time in enumerate(minute_times_list):
                        if minute_time not in processed_times:
                            start_index = i
                            break
                    else:
                        # Все свечи уже обработаны
                        start_index = len(minute_times_list)
                    
                    if start_index < len(minute_times_list):
                        print(f"  [{start_timestamp}] ✓ Найден сохраненный прогресс: обработано {len(processed_times)} свечей")
                        print(f"  [{start_timestamp}] Продолжаем с позиции {start_index + 1}/{len(minute_times)}")
                    else:
                        print(f"  [{start_timestamp}] ✓ Все свечи уже обработаны в сохраненном прогрессе")
                        return ticks_data
        
        loaded_count = len(ticks_data)
        empty_count = 0
        cached_count = 0
        last_save_index = start_index
        
        print(f"  [{start_timestamp}] Загрузка тиков для {len(minute_times)} минутных свечей...")
        if start_index > 0:
            print(f"  [{start_timestamp}] Пропускаем {start_index} уже обработанных свечей")
        
        for i in range(start_index, len(minute_times)):
            minute_time = minute_times[i]
            
            # Пропускаем уже обработанные
            if minute_time in processed_times:
                continue
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                current_timestamp = self._get_timestamp()
                duration_str = self._format_duration(elapsed)
                print(f"    [{current_timestamp}] Обработано {i + 1}/{len(minute_times)} свечей... (время работы: {duration_str})")
            
            try:
                ticks = self.load_ticks_for_minute(symbol, minute_time, lookback_minutes)
                if not ticks.empty:
                    ticks_data[minute_time] = ticks
                    processed_times.add(minute_time)
                    dirty_minutes.append(minute_time)
                    loaded_count += 1
                    
                    # Диагностика для первых 5 загруженных тиков
                    if loaded_count <= 5:
                        time_range_start = minute_time - timedelta(minutes=lookback_minutes)
                        print(f"    [{self._get_timestamp()}] Пример {loaded_count}: Минута {minute_time}, тиков: {len(ticks)}, диапазон: {time_range_start} - {minute_time}")
                    
                    # Проверяем, были ли тики из кэша
                    if self.use_cache and self.cache is not None:
                        cached = self.cache.get_cached_ticks(
                            symbol,
                            minute_time - timedelta(minutes=lookback_minutes),
                            minute_time
                        )
                        if cached is not None and not cached.empty:
                            cached_count += 1
                else:
                    empty_count += 1
                    # Диагностика для первых 5 пустых тиков
                    if empty_count <= 5:
                        time_range_start = minute_time - timedelta(minutes=lookback_minutes)
                        print(f"    [{self._get_timestamp()}] Пустой {empty_count}: Минута {minute_time}, диапазон: {time_range_start} - {minute_time}")
            except Exception as e:
                # Пропускаем ошибки для отдельных свечей
                empty_count += 1
                if (i + 1) % 1000 == 0:
                    print(f"    Предупреждение при загрузке тиков для {minute_time}: {e}")
            
            # Сохраняем прогресс периодически
            if self.use_cache and (i + 1) % save_progress_every == 0:
                try:
                    elapsed = time.time() - start_time
                    current_timestamp = self._get_timestamp()
                    duration_str = self._format_duration(elapsed)
                    self._save_batch_progress(
                        symbol,
                        ticks_data,
                        list(processed_times),
                        dirty_minutes=dirty_minutes
                    )
                    dirty_minutes.clear()
                    print(f"    [{current_timestamp}] ✓ Прогресс сохранен ({i + 1}/{len(minute_times)} свечей, время работы: {duration_str})")
                    last_save_index = i + 1
                except Exception as e:
                    current_timestamp = self._get_timestamp()
                    print(f"    [{current_timestamp}] ⚠ Ошибка при сохранении прогресса: {e}")
        
        # Сохраняем финальный прогресс
        if self.use_cache and len(ticks_data) > 0 and dirty_minutes:
            try:
                self._save_batch_progress(
                    symbol,
                    ticks_data,
                    list(processed_times),
                    dirty_minutes=dirty_minutes
                )
                dirty_minutes.clear()
            except Exception as e:
                current_timestamp = self._get_timestamp()
                print(f"    [{current_timestamp}] ⚠ Ошибка при финальном сохранении прогресса: {e}")
        
        # Выводим итоговую статистику
        total_elapsed = time.time() - start_time
        end_timestamp = self._get_timestamp()
        total_duration_str = self._format_duration(total_elapsed)
        print(f"  [{end_timestamp}] Загружено тиков для {loaded_count} свечей (из кэша: {cached_count}), пустых: {empty_count}")
        print(f"  [{end_timestamp}] Общее время обработки: {total_duration_str}")
        
        # Если все обработано успешно, удаляем файл прогресса
        if len(ticks_data) == len(minute_times):
            self._clear_batch_progress(symbol)
            print(f"  [{end_timestamp}] ✓ Обработка завершена, файл прогресса удален")
        
        return ticks_data
    
    def disconnect(self):
        """Отключение от MT5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False

