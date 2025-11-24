"""
Модуль для загрузки исторических данных из MetaTrader 5
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
try:
    import MetaTrader5 as mt5
except ImportError as e:
    mt5 = None
    MT5_IMPORT_ERROR = e

class MT5DataLoader:
    """Класс для загрузки данных из MetaTrader 5"""
    
    def __init__(self, login: Optional[int] = None, password: Optional[str] = None, 
                 server: Optional[str] = None):
        """
        Инициализация подключения к MT5
        
        Args:
            login: Логин для подключения (если None, используется автоподключение)
            password: Пароль для подключения
            server: Сервер брокера
        """
        self.login = login
        self.password = password
        self.server = server
        self.connected = False
        
    def connect(self) -> bool:
        """
        Подключение к MetaTrader 5
        
        Returns:
            True если подключение успешно, False иначе
        """
        if mt5 is None:
            raise ImportError(
                "Библиотека MetaTrader5 не установлена или недоступна. "
                "Установите пакет `MetaTrader5` и убедитесь, что терминал MT5 запущен."
            ) from MT5_IMPORT_ERROR

        if not mt5.initialize():
            print("Ошибка инициализации MT5")
            return False
        
        if self.login is not None:
            if not mt5.login(self.login, password=self.password, server=self.server):
                print(f"Ошибка подключения к MT5. Логин: {self.login}")
                mt5.shutdown()
                return False
        
        self.connected = True
        print("Успешное подключение к MT5")
        return True
    
    def disconnect(self):
        """Отключение от MetaTrader 5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("Отключение от MT5")
    
    def get_symbols(self) -> List[str]:
        """
        Получить список доступных символов
        
        Returns:
            Список символов
        """
        if not self.connected:
            if not self.connect():
                return []
        
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        
        return [s.name for s in symbols]
    
    def load_data(self, symbol: str, timeframe: str, start_date: datetime, 
                  end_date: Optional[datetime] = None, count: Optional[int] = None) -> pd.DataFrame:
        """
        Загрузка исторических данных
        
        Args:
            symbol: Символ валютной пары (например, 'EURUSD')
            timeframe: Таймфрейм ('M1', 'M5', 'H1', 'H4', 'D1' и т.д.)
            start_date: Начальная дата
            end_date: Конечная дата (если None, загружаются данные до текущего момента)
            count: Количество баров (если указано, имеет приоритет над датами)
        
        Returns:
            DataFrame с колонками: time, open, high, low, close, tick_volume, spread, real_volume
        """
        if not self.connected:
            if not self.connect():
                return pd.DataFrame()
        
        # Преобразование таймфрейма
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe.upper())
        if mt5_timeframe is None:
            print(f"Неизвестный таймфрейм: {timeframe}")
            return pd.DataFrame()
        
        # Проверка и активация символа
        if not mt5.symbol_select(symbol, True):
            error_info = mt5.last_error()
            print(f"⚠️  Ошибка: Символ {symbol} не найден или не может быть активирован")
            if error_info:
                if isinstance(error_info, tuple):
                    print(f"   Код ошибки MT5: {error_info[0]}")
                    print(f"   Описание: {error_info[1]}")
                else:
                    print(f"   Код ошибки MT5: {error_info}")
            
            # Попробуем найти похожие символы
            all_symbols = self.get_symbols()
            if all_symbols:
                similar = [s for s in all_symbols if 'GOLD' in s.upper() or 'XAU' in s.upper() or 'AU' in s.upper() or symbol.upper() in s.upper()]
                if similar:
                    print(f"   Похожие символы: {', '.join(similar[:10])}")
                else:
                    print(f"   Всего доступно символов: {len(all_symbols)}")
                    print(f"   Примеры: {', '.join(all_symbols[:10])}")
            return pd.DataFrame()
        
        # Загрузка данных
        if count is not None:
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        else:
            if end_date is None:
                end_date = datetime.now()
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            error_info = mt5.last_error()
            print(f"⚠️  Ошибка: Не удалось загрузить данные для {symbol}")
            print(f"   Период: {start_date.strftime('%Y-%m-%d %H:%M')} - {end_date.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Таймфрейм: {timeframe}")
            if error_info:
                if isinstance(error_info, tuple):
                    print(f"   Код ошибки MT5: {error_info[0]}")
                    print(f"   Описание: {error_info[1]}")
                else:
                    print(f"   Код ошибки MT5: {error_info}")
            
            # Проверяем, есть ли вообще данные для символа
            test_rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, 1)
            if test_rates is None or len(test_rates) == 0:
                print(f"   ⚠️  Для символа {symbol} нет данных даже на текущий момент")
                print(f"   Проверьте, что символ доступен в вашем брокере и добавлен в Market Watch")
            else:
                print(f"   ℹ️  Данные для символа есть, но не за указанный период")
                print(f"   Попробуйте:")
                print(f"     - Уменьшить период (меньше месяцев)")
                print(f"     - Проверить доступность исторических данных у брокера")
                print(f"     - Убедиться, что терминал загрузил историю для символа")
            return pd.DataFrame()
        
        # Преобразование в DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Переименование колонок для удобства
        df.rename(columns={
            'tick_volume': 'volume',
            'real_volume': 'real_volume'
        }, inplace=True)
        
        return df
    
    def load_multiple_timeframes(self, symbol: str, timeframes: List[str], 
                                 start_date: datetime, 
                                 end_date: Optional[datetime] = None) -> dict:
        """
        Загрузка данных для нескольких таймфреймов
        
        Args:
            symbol: Символ валютной пары
            timeframes: Список таймфреймов
            start_date: Начальная дата
            end_date: Конечная дата
        
        Returns:
            Словарь {timeframe: DataFrame}
        """
        data = {}
        for tf in timeframes:
            df = self.load_data(symbol, tf, start_date, end_date)
            if not df.empty:
                data[tf] = df
        return data
    
    def __enter__(self):
        """Контекстный менеджер - вход"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер - выход"""
        self.disconnect()
    
    @staticmethod
    def create_minute_candles_from_ticks(ticks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает минутные свечи из тиковых данных.
        Все цены (OHLC) строятся только по bid, как в MT5.
        
        Args:
            ticks_df: DataFrame с тиками (индекс - время, колонки: bid, ask, volume, spread)
        
        Returns:
            DataFrame с минутными свечами (open, high, low, close, volume, tick_volume)
            Структура соответствует стандартным OHLC данным MT5
        """
        if ticks_df.empty:
            return pd.DataFrame()
        
        # Ресэмплинг по минутам
        rule = '1T'  # 1 минута
        
        # Все цены строятся только по bid, как в MT5
        open_price = ticks_df['bid'].resample(rule).first()   # Open - первая bid
        high = ticks_df['bid'].resample(rule).max()           # High - максимальный bid
        low = ticks_df['bid'].resample(rule).min()            # Low - минимальный bid
        close = ticks_df['bid'].resample(rule).last()         # Close - последняя bid
        
        # Объемы
        tick_volume = ticks_df['bid'].resample(rule).count()  # Количество тиков
        volume = ticks_df['volume'].resample(rule).sum() if 'volume' in ticks_df.columns else tick_volume
        
        # Спред (для информации, не используется в OHLC)
        spread = ticks_df['spread'].resample(rule).mean() if 'spread' in ticks_df.columns else pd.Series(dtype=float)
        
        # Создание DataFrame
        candles = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'tick_volume': tick_volume,
            'volume': volume,
            'spread': spread
        })
        
        # Переименование для совместимости с MT5 форматом
        candles.rename(columns={
            'tick_volume': 'volume',
            'volume': 'real_volume'
        }, inplace=True)
        
        # Удаление строк с NaN (минуты без тиков)
        candles = candles.dropna(subset=['open', 'high', 'low', 'close'])
        
        return candles

