<!-- bf94ef84-daa9-43a9-b93b-e24db14fe504 df0cd12b-2758-4b58-83f1-93c3508f940e -->
# Реорганизация проекта и улучшение стратегии

## 1. Реорганизация структуры проекта в Workspace

### 1.1. Создать новую структуру директорий

```
workspace/
  ├── raw_data/          # Исходные данные и кэши
  │   ├── ticks/        # Тиковые данные
  │   └── cache/        # Кэши подготовленных данных
  ├── prepared/         # Готовые данные для обучения
  │   ├── features/     # CSV файлы с фичами (gold_train.csv, gold_val.csv, gold_test.csv)
  │   └── scalers/      # Feature scalers
  ├── models/           # Обученные модели и метрики
  │   ├── checkpoints/  # Модели (.pth)
  │   ├── metrics/      # Метрики обучения (history CSV, графики, confusion matrix)
  │   └── logs/         # TensorBoard логи
  └── results/          # Результаты бэктестов
      ├── backtests/    # Результаты бэктестинга
      └── monitoring/   # Графики мониторинга
```

### 1.2. Обновить пути в коде

**Файлы для изменения:**

- `data/gold_data_prep.py`: `cache_dir='data/prepared'` → `cache_dir='workspace/raw_data/cache'`
- `data/tick_data_loader.py`: пути к тикам `'data/ticks'` → `'workspace/raw_data/ticks'`
- `prepare_gold_data.py`: пути сохранения CSV `'data/gold_*.csv'` → `'workspace/prepared/features/gold_*.csv'`
- `models/trainer.py`: 
  - пути сохранения моделей `'models/checkpoints'` → `'workspace/models/checkpoints'`
  - пути сохранения scaler `'models/feature_scaler_*.pkl'` → `'workspace/prepared/scalers/feature_scaler_*.pkl'`
  - пути сохранения метрик `'models/checkpoints/*_history.csv'` → `'workspace/models/metrics/*_history.csv'`
  - пути сохранения графиков `'models/checkpoints/*_training_curves.png'` → `'workspace/models/metrics/*_training_curves.png'`
  - пути сохранения confusion matrix `'models/confusion_matrix_*.png'` → `'workspace/models/metrics/confusion_matrix_*.png'`
- `train_model.py`, `train_all_models.py`: обновить все пути к моделям и scalers
- `trading/backtester.py`: обновить пути загрузки моделей и scalers
- `backtest_strategy.py`: обновить пути сохранения результатов в `workspace/results/`
- `trading/monitoring_visualizer.py`: обновить пути сохранения графиков в `workspace/results/monitoring/`

### 1.3. Обновить .gitignore

- Добавить `workspace/` в `.gitignore`
- Старые данные в `data/ticks/`, `data/prepared/`, `models/checkpoints/` можно удалить

## 2. Изменение стратегии на 5 классов с отложенным входом

### 2.1. Изменение генерации таргетов

**Файл: `data/target_generator.py`**

**Изменения:**

1. Изменить `classify_signal()` для генерации 5 классов:

   - 0 = Неопределенность
   - 1 = Пробой вверх (BUY)
   - 2 = Пробой вниз (SELL)
   - 3 = Отскок вверх (BUY после падения)
   - 4 = Отскок вниз (SELL после роста)

2. Увеличить пороги:

   - `breakout_threshold=200.0` (вместо 50.0)
   - `bounce_threshold=150.0` (вместо 30.0)

3. Изменить логику проверки:

   - Сначала проверять отскок (приоритет), затем пробой
   - Улучшить `_check_bounce()`: проверять, что разворот произошел до 70% периода

**Файл: `data/gold_data_prep.py`**

- Обновить параметры TargetGenerator: `breakout_threshold=200.0`, `bounce_threshold=150.0`

### 2.2. Обновление конфигурации модели

**Файлы:**

- `config/model_config.py`: изменить `num_classes=5` (вместо 3)
- `train_model.py`, `train_all_models.py`: обновить создание конфигурации

### 2.3. Реализация отложенного входа для отскоков

**Файл: `trading/backtester.py`**

**Изменения:**

1. В `__init__`: добавить `self.pending_bounce_signals: dict = {}`

2. В `get_signal()`:

   - Для классов 1, 2 (пробой) - возвращать сигнал сразу
   - Для классов 3, 4 (отскок) - сохранять в `pending_bounce_signals` и возвращать `None`

3. Добавить метод `check_bounce_confirmation()`:

   - Проверять разворот по паттерну свечей
   - Для отскока вверх: минимум за последние 3-5 свечей, затем рост
   - Для отскока вниз: максимум за последние 3-5 свечей, затем падение
   - Возвращать `(direction, confidence)` или `None`

4. В `backtest()`: перед получением новых сигналов проверять подтверждение отложенных отскоков

**Файл: `config/trading_config.py`**

Добавить параметры:

- `bounce_confirmation_enabled: bool = True`
- `bounce_confirmation_periods: int = 3`
- `bounce_min_reversal_pips: float = 10.0`

### 2.4. Обновление документации

**Файлы:**

- `docs/03_DATA_PREPARATION.md` - обновить описание классов (5 вместо 3)
- `docs/06_BACKTESTING_GUIDE.md` - описать логику отложенного входа для отскоков
- `docs/10_TRADING_MODEL_PLAN.md` - обновить план с учетом новых классов

## 3. Расширение paperspace_utils.py

### 3.1. Переименование и рефакторинг

- Переименовать `upload_to_paperspace.py` → `paperspace_utils.py`
- Реорганизовать в классы: `PaperspaceUploader`, `PaperspaceDownloader`

### 3.2. Функционал загрузки данных для обучения

**Класс: `PaperspaceUploader`**

**Методы:**

- `create_training_archive()` - упаковка данных для обучения:
  - `workspace/prepared/features/*.csv` - CSV файлы
  - `workspace/raw_data/cache/` - кэши (опционально)
  - `workspace/raw_data/ticks/` - тики (опционально, с подтверждением)

- `upload_training_data()` - загрузка через SCP/RSYNC

### 3.3. Функционал загрузки результатов обучения

**Класс: `PaperspaceDownloader`**

**Методы:**

- `create_results_archive()` - упаковка результатов обучения:
  - `workspace/models/checkpoints/*.pth` - модели
  - `workspace/models/metrics/*.csv`, `*.png` - метрики
  - `workspace/prepared/scalers/*.pkl` - scalers

- `download_results()` - скачивание результатов с Paperspace через SCP/RSYNC
- `list_remote_files()` - список файлов на Paperspace

### 3.4. CLI интерфейс

**Команды:**

- `python paperspace_utils.py upload-training [--host HOST] [--path PATH]` - загрузить данные для обучения
- `python paperspace_utils.py download-results [--host HOST] [--path PATH]` - скачать результаты обучения
- `python paperspace_utils.py create-training-archive [--output FILE]` - только создать архив
- `python paperspace_utils.py create-results-archive [--output FILE]` - только создать архив результатов

**Файлы:**

- `paperspace_utils.py` - новый файл (переименовать и расширить)
- `docs/13_PAPERSPACE_SETUP.md` - обновить документацию

## 4. Добавление интерактивных метрик

### 4.1. Интеграция TensorBoard

**Файл: `models/trainer.py`**

**Изменения:**

1. Добавить импорт: `from torch.utils.tensorboard import SummaryWriter`

2. В `__init__`: создать `self.writer = SummaryWriter(log_dir=f'workspace/models/logs/{model_type}_{timestamp}')`

3. В `train_epoch()`: логировать метрики по батчам:

   - `self.writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)`
   - `self.writer.add_scalar('Accuracy/Train_Batch', accuracy, global_step)`

4. В `train()`: логировать метрики по эпохам:

   - `self.writer.add_scalar('Loss/Train', train_loss, epoch)`
   - `self.writer.add_scalar('Loss/Val', val_loss, epoch)`
   - `self.writer.add_scalar('Accuracy/Train', train_acc, epoch)`
   - `self.writer.add_scalar('Accuracy/Val', val_acc, epoch)`
   - `self.writer.add_scalar('Learning_Rate', lr, epoch)`

5. В конце обучения: `self.writer.close()`

**Файлы:**

- `models/trainer.py` - интеграция TensorBoard
- `requirements.txt`, `requirements_linux.txt` - добавить `tensorboard>=2.10.0`

### 4.2. Интеграция Weights & Biases (wandb)

**Файл: `models/trainer.py`**

**Изменения:**

1. Добавить опциональные параметры в `__init__`:

   - `use_wandb: bool = False`
   - `wandb_project: str = "xauusd-ai-ea"`

2. При `use_wandb=True`: инициализировать `wandb.init(project=wandb_project, name=f"{model_type}_{timestamp}")`

3. Логировать те же метрики что и в TensorBoard:

   - `wandb.log({'train_loss': train_loss, 'val_loss': val_loss, 'train_acc': train_acc, 'val_acc': val_acc, 'lr': lr}, step=epoch)`

4. В конце обучения: `wandb.finish()`

**Файлы:**

- `models/trainer.py` - интеграция wandb
- `train_model.py`, `train_all_models.py` - добавить CLI аргументы `--use-wandb` и `--wandb-project`
- `requirements.txt`, `requirements_linux.txt` - добавить `wandb>=0.15.0`

### 4.3. Скрипт запуска TensorBoard

**Файл: `scripts/start_tensorboard.py`**

**Функционал:**

- Автоматически находить последний log директорий в `workspace/models/logs/`
- Запускать TensorBoard на порту 6006
- Поддержка SSH туннеля для удаленного доступа
- Вывод команды для SSH туннеля

### 4.4. Документация по метрикам

**Файл: `docs/14_METRICS_GUIDE.md`**

**Содержание:**

- Установка и настройка TensorBoard
- Установка и настройка W&B
- Запуск TensorBoard
- Доступ через SSH туннель
- Интерпретация метрик
- Сравнение экспериментов
- Рекомендации по выбору инструмента

**Файлы:**

- `docs/14_METRICS_GUIDE.md` - новый файл
- `docs/00_INDEX.md` - добавить ссылку

## 5. Памятка о следующих шагах

**Файл: `.cursor/plans/NEXT_STEPS.md`**

**Содержание:**

- Проверка распределения классов (должно быть ~60-70% класс 0, ~15-20% пробои, ~5-10% отскоки)
- Проверка на data leakage (корреляция фичей с таргетом)
- Валидация генерации таргетов (визуализация примеров)
- Тестирование на малой выборке (1-2 месяца) перед полным обучением
- Проверка метрик на валидационной выборке (accuracy, precision, recall по классам)
- Анализ confusion matrix для выявления проблемных классов
- Рекомендации по настройке порогов если распределение неправильное

**Примечание:** НЕ добавлять ссылку в `docs/00_INDEX.md`

## 6. Обновление зависимостей и документации

### 6.1. Обновить requirements

**Файлы:**

- `requirements.txt` - добавить `tensorboard>=2.10.0` и `wandb>=0.15.0`
- `requirements_linux.txt` - добавить `tensorboard>=2.10.0` и `wandb>=0.15.0`

### 6.2. Обновить CHANGELOG.md

**Добавить записи:**

- Реорганизация структуры в workspace/
- Переход на 5 классов (пробой/отскок с направлениями)
- Отложенный вход для отскоков
- Интеграция TensorBoard и W&B
- Расширение paperspace_utils.py

### 6.3. Обновить README.md

**Изменения:**

- Обновить структуру проекта
- Добавить информацию о Workspace
- Обновить инструкции по использованию

## Порядок выполнения

1. Создать структуру `workspace/` (старые данные можно удалить)
2. Обновить все пути в коде на новую структуру
3. Изменить генерацию таргетов на 5 классов
4. Реализовать отложенный вход для отскоков
5. Расширить paperspace_utils.py
6. Интегрировать TensorBoard и W&B
7. Создать документацию и памятку
8. Обновить CHANGELOG и README

### To-dos

- [ ] Создать структуру workspace/ и скрипт миграции данных
- [ ] Обновить все пути в коде на новую структуру workspace/
- [ ] Изменить генерацию таргетов на 5 классов с улучшенной логикой
- [ ] Реализовать отложенный вход для отскоков в backtester
- [ ] Переименовать и расширить upload_to_paperspace.py в paperspace_utils.py
- [ ] Интегрировать TensorBoard в процесс обучения
- [ ] Создать документацию по TensorBoard и памятку о следующих шагах
- [ ] Обновить CHANGELOG.md и README.md с новыми изменениями