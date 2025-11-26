"""
Скрипт для визуализации данных обучения
Отображает минутные свечи с сигналами и точками закрытия
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
from datetime import datetime, timedelta

# Цвета для разных классов сигналов
SIGNAL_COLORS = {
    0: 'gray',      # Неопределенность
    1: 'green',    # Пробой вверх (BUY)
    2: 'red',      # Пробой вниз (SELL)
    3: 'lime',     # Отскок вверх (BUY после падения)
    4: 'orange'    # Отскок вниз (SELL после роста)
}

SIGNAL_LABELS = {
    0: 'Неопределенность',
    1: 'Пробой вверх (BUY)',
    2: 'Пробой вниз (SELL)',
    3: 'Отскок вверх (BUY)',
    4: 'Отскок вниз (SELL)'
}

SIGNAL_MARKERS = {
    0: None,
    1: '^',  # Треугольник вверх для покупки
    2: 'v',  # Треугольник вниз для продажи
    3: '^',  # Треугольник вверх для отскока вверх
    4: 'v'   # Треугольник вниз для отскока вниз
}


def load_training_data(file_path: str = 'workspace/prepared/features/gold_train.csv') -> pd.DataFrame:
    """
    Загружает данные обучения
    
    Args:
        file_path: Путь к файлу с данными обучения
    
    Returns:
        DataFrame с данными
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}\nСначала запустите: python prepare_gold_data.py")
    
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Загружено {len(df)} строк из {file_path}")
    return df


def filter_by_days(df: pd.DataFrame, days: int = 5, start_date: str = None) -> pd.DataFrame:
    """
    Фильтрует данные по количеству дней
    
    Args:
        df: DataFrame с данными
        days: Количество дней для отображения
        start_date: Начальная дата (если None, берется последняя дата - days)
    
    Returns:
        Отфильтрованный DataFrame
    """
    if start_date:
        start = pd.to_datetime(start_date)
    else:
        # Берем последние N дней
        start = df.index[-1] - timedelta(days=days)
    
    end = start + timedelta(days=days)
    filtered = df[(df.index >= start) & (df.index <= end)].copy()
    print(f"Отфильтровано {len(filtered)} строк за период {start.date()} - {end.date()}")
    return filtered


def find_exit_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит точки закрытия сигналов на основе future_return
    
    Args:
        df: DataFrame с данными
    
    Returns:
        DataFrame с добавленными колонками exit_index и exit_price
    """
    df = df.copy()
    df['exit_index'] = None
    df['exit_price'] = None
    df['exit_period'] = None
    
    # Проверяем наличие колонок future_return
    future_return_cols = [col for col in df.columns if col.startswith('future_return_')]
    if not future_return_cols:
        print("⚠️  Колонки future_return_* не найдены. Точки закрытия не будут отображены.")
        return df
    
    for i in range(len(df)):
        signal_class = df['signal_class'].iloc[i]
        
        # Пропускаем неопределенность
        if signal_class == 0 or pd.isna(signal_class):
            continue
        
        # Находим максимальную доходность в будущем
        max_return = -np.inf
        max_period = None
        max_return_idx = None
        
        for col in future_return_cols:
            period = int(col.split('_')[-1])
            future_idx = i + period
            
            if future_idx < len(df):
                return_val = df[col].iloc[i]
                if not pd.isna(return_val):
                    # Для пробоев ищем максимальную абсолютную доходность
                    # Для отскоков учитываем знак
                    if signal_class in [1, 2]:  # Пробои
                        if abs(return_val) > abs(max_return):
                            max_return = return_val
                            max_period = period
                            max_return_idx = future_idx
                    else:  # Отскоки
                        # Для отскока вверх ищем положительную доходность
                        # Для отскока вниз - отрицательную
                        if (signal_class == 3 and return_val > 0 and return_val > max_return) or \
                           (signal_class == 4 and return_val < 0 and return_val < max_return):
                            max_return = return_val
                            max_period = period
                            max_return_idx = future_idx
                        elif abs(return_val) > abs(max_return):
                            max_return = return_val
                            max_period = period
                            max_return_idx = future_idx
        
        if max_return_idx is not None:
            df.loc[df.index[i], 'exit_index'] = max_return_idx
            df.loc[df.index[i], 'exit_price'] = df['close'].iloc[max_return_idx]
            df.loc[df.index[i], 'exit_period'] = max_period
    
    return df


def plot_candlestick(ax, df: pd.DataFrame, width: float = 0.6):
    """
    Рисует свечной график
    
    Args:
        ax: Axes для отрисовки
        df: DataFrame с данными (open, high, low, close)
        width: Ширина свечи
    """
    # Цвета свечей
    colors = ['green' if close >= open else 'red' 
              for open, close in zip(df['open'], df['close'])]
    
    for i, (idx, row) in enumerate(df.iterrows()):
        open_price = row['open']
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']
        
        # Тело свечи
        body_low = min(open_price, close_price)
        body_high = max(open_price, close_price)
        body_height = body_high - body_low
        
        # Тени
        upper_shadow = high_price - body_high
        lower_shadow = body_low - low_price
        
        # Рисуем тело
        if body_height > 0:
            rect = Rectangle(
                (i - width/2, body_low),
                width,
                body_height,
                facecolor=colors[i],
                edgecolor='black',
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(rect)
        
        # Рисуем тени
        ax.plot([i, i], [low_price, high_price], 
                color='black', linewidth=0.5, alpha=0.5)


def visualize_signals(df: pd.DataFrame, 
                     output_path: str = None,
                     show_uncertainty: bool = False):
    """
    Визуализирует сигналы на свечном графике
    
    Args:
        df: DataFrame с данными
        output_path: Путь для сохранения графика
        show_uncertainty: Показывать ли неопределенность (класс 0)
    """
    # Находим точки закрытия
    df = find_exit_points(df)
    
    # Фильтруем неопределенность, если нужно
    if not show_uncertainty:
        df_signals = df[df['signal_class'] != 0].copy()
    else:
        df_signals = df.copy()
    
    # Создаем фигуру
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Рисуем свечи
    plot_candlestick(ax, df)
    
    # Отмечаем сигналы
    for signal_class in [1, 2, 3, 4]:
        signal_data = df_signals[df_signals['signal_class'] == signal_class]
        if len(signal_data) == 0:
            continue
        
        # Получаем индексы для отображения
        signal_indices = [df.index.get_loc(idx) for idx in signal_data.index]
        
        # Рисуем маркеры сигналов
        marker = SIGNAL_MARKERS[signal_class]
        color = SIGNAL_COLORS[signal_class]
        label = SIGNAL_LABELS[signal_class]
        
        if marker:
            ax.scatter(signal_indices, 
                      signal_data['close'].values,
                      marker=marker,
                      color=color,
                      s=200,
                      edgecolors='black',
                      linewidths=1.5,
                      label=label,
                      zorder=5)
    
    # Рисуем линии к точкам закрытия
    for idx, row in df_signals.iterrows():
        if pd.notna(row.get('exit_index')) and row['exit_index'] < len(df):
            signal_idx = df.index.get_loc(idx)
            exit_idx = int(row['exit_index'])
            signal_price = row['close']
            exit_price = row['exit_price']
            
            # Цвет линии зависит от типа сигнала
            signal_class = int(row['signal_class'])
            color = SIGNAL_COLORS[signal_class]
            
            # Рисуем линию от сигнала к точке закрытия
            ax.plot([signal_idx, exit_idx],
                   [signal_price, exit_price],
                   color=color,
                   linestyle='--',
                   alpha=0.5,
                   linewidth=1.5)
            
            # Рисуем точку закрытия
            ax.scatter(exit_idx,
                      exit_price,
                      marker='x',
                      color=color,
                      s=150,
                      linewidths=2,
                      zorder=6)
    
    # Настройка осей
    ax.set_xlabel('Время', fontsize=12)
    ax.set_ylabel('Цена (XAUUSD)', fontsize=12)
    ax.set_title(f'Визуализация сигналов обучения\n'
                f'Период: {df.index[0].strftime("%Y-%m-%d")} - {df.index[-1].strftime("%Y-%m-%d")}',
                fontsize=14, fontweight='bold')
    
    # Устанавливаем метки на оси X
    ax.set_xticks(range(0, len(df), max(1, len(df) // 10)))
    ax.set_xticklabels([df.index[i].strftime('%Y-%m-%d\n%H:%M') 
                       for i in range(0, len(df), max(1, len(df) // 10))],
                      rotation=45, ha='right')
    
    # Легенда
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Сетка
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ График сохранен: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Визуализация данных обучения с сигналами и точками закрытия',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python utils/visualize_training_data.py                    # Последние 5 дней
  python utils/visualize_training_data.py --days 10          # Последние 10 дней
  python utils/visualize_training_data.py --start-date 2024-01-01  # С конкретной даты
  python utils/visualize_training_data.py --show-uncertainty       # Показать неопределенность
  python utils/visualize_training_data.py --output workspace/results/training_visualization.png
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='workspace/prepared/features/gold_train.csv',
        help='Путь к файлу с данными обучения (по умолчанию: workspace/prepared/features/gold_train.csv)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=5,
        help='Количество дней для отображения (по умолчанию: 5)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: последние N дней)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь для сохранения графика (по умолчанию: показывается на экране)'
    )
    
    parser.add_argument(
        '--show-uncertainty',
        action='store_true',
        help='Показывать неопределенность (класс 0) на графике'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ ДАННЫХ ОБУЧЕНИЯ")
    print("=" * 80)
    
    # Загрузка данных
    print(f"\n1. Загрузка данных из {args.input}...")
    try:
        df = load_training_data(args.input)
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        return 1
    
    # Проверка наличия необходимых колонок
    required_cols = ['open', 'high', 'low', 'close', 'signal_class']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Ошибка: Отсутствуют необходимые колонки: {missing_cols}")
        return 1
    
    # Фильтрация по дням
    print(f"\n2. Фильтрация данных ({args.days} дней)...")
    df_filtered = filter_by_days(df, days=args.days, start_date=args.start_date)
    
    if len(df_filtered) == 0:
        print("❌ Ошибка: Нет данных за указанный период")
        return 1
    
    # Статистика сигналов
    print(f"\n3. Статистика сигналов в выбранном периоде:")
    signal_counts = df_filtered['signal_class'].value_counts().sort_index()
    for signal_class, count in signal_counts.items():
        label = SIGNAL_LABELS.get(int(signal_class), f'Класс {signal_class}')
        print(f"   {label}: {count} ({count/len(df_filtered)*100:.1f}%)")
    
    # Визуализация
    print(f"\n4. Создание визуализации...")
    output_path = args.output
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    visualize_signals(df_filtered, output_path=output_path, show_uncertainty=args.show_uncertainty)
    
    print("\n" + "=" * 80)
    print("Визуализация завершена!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    exit(main())

