"""
Оптимизированная версия загрузчика тиков с возможностью прерывания
"""
import signal
import sys
from data.tick_data_loader import TickDataLoader

class InterruptibleTickLoader(TickDataLoader):
    """Загрузчик тиков с возможностью прерывания (Ctrl+C)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигнала прерывания"""
        print("\n\n⚠ Прерывание загрузки тиков...")
        print("Сохраняем уже загруженные данные...")
        self.interrupted = True
    
    def ensure_cache_coverage(self, symbol: str, required_end: datetime):
        """Переопределяем с поддержкой прерывания"""
        if not self.use_cache or self.cache is None:
            return
        
        required_start = required_end - timedelta(days=self.default_lookback_days)
        needs_update, missing_start, missing_end = self.cache.needs_update(
            symbol, required_start, required_end
        )
        
        if not needs_update:
            print(f"  Кэш тиков для {symbol} актуален")
            return
        
        print(f"  Обновление кэша тиков для {symbol}...")
        print(f"    Требуемый диапазон: {missing_start} - {missing_end}")
        print(f"    Нажмите Ctrl+C для прерывания (данные сохранятся)")
        
        self._ensure_connected()
        
        if not mt5.symbol_select(symbol, True):
            print(f"  Предупреждение: символ {symbol} не найден")
            return
        
        # Загружаем с проверкой прерывания
        current_date = missing_start
        loaded_days = 0
        total_ticks = 0
        all_ticks_list = []
        
        days_to_load = (missing_end - missing_start).days + 1
        if days_to_load > 30:
            print(f"    ⚠ Загружаем только последние 30 дней")
            current_date = max(missing_start, missing_end - timedelta(days=30))
            days_to_load = 30
        
        try:
            while current_date <= missing_end and not self.interrupted:
                day_end = min(current_date + timedelta(days=1), missing_end)
                
                try:
                    ticks = mt5.copy_ticks_range(symbol, current_date, day_end, mt5.COPY_TICKS_ALL)
                    
                    if ticks is not None and len(ticks) > 0:
                        df = pd.DataFrame(ticks)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('time', inplace=True)
                        mask = (df.index >= current_date) & (df.index <= day_end)
                        df = df[mask]
                        
                        if not df.empty:
                            df['spread'] = df['ask'] - df['bid']
                            df['spread_pips'] = df['spread'] * 10000
                            all_ticks_list.append(df)
                            loaded_days += 1
                            total_ticks += len(df)
                    
                    # Сохраняем каждые 3 дня
                    if len(all_ticks_list) >= 3:
                        combined_df = pd.concat(all_ticks_list)
                        self.cache.save_ticks(symbol, combined_df, batch_size=3)
                        all_ticks_list = []
                        print(f"    ✓ Сохранено {loaded_days} дней, тиков: {total_ticks:,}")
                
                except Exception as e:
                    if not self.interrupted:
                        print(f"    Ошибка за {current_date.date()}: {e}")
                
                current_date = day_end
            
            # Сохраняем оставшиеся
            if all_ticks_list and not self.interrupted:
                combined_df = pd.concat(all_ticks_list)
                self.cache.save_ticks(symbol, combined_df, batch_size=3)
            
            if self.interrupted:
                print(f"\n  ⚠ Загрузка прервана пользователем")
                print(f"  ✓ Сохранено {loaded_days} дней, тиков: {total_ticks:,}")
            else:
                print(f"  ✓ Кэш обновлен: {loaded_days} дней, тиков: {total_ticks:,}")
        
        except KeyboardInterrupt:
            print(f"\n  ⚠ Прерывание...")
            if all_ticks_list:
                combined_df = pd.concat(all_ticks_list)
                self.cache.save_ticks(symbol, combined_df, batch_size=3)
            print(f"  ✓ Сохранено {loaded_days} дней")

