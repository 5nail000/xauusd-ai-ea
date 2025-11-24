"""
Модуль для загрузки исторических данных из MetaTrader 5
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import MetaTrader5 as mt5

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
        
        # Загрузка данных
        if count is not None:
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
        else:
            if end_date is None:
                end_date = datetime.now()
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"Не удалось загрузить данные для {symbol}")
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

