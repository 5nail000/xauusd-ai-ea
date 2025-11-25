# Настройка обучения на Paperspace

## Обзор

Это руководство поможет вам перенести подготовленные данные на Paperspace и запустить обучение модели там, где доступно больше ресурсов (30 ГБ RAM).

## Преимущества Paperspace

- ✅ 30 ГБ оперативной памяти (достаточно для обучения на 12+ месяцах данных)
- ✅ GPU ускорение (опционально)
- ✅ Не требует MT5 (обучение работает только с CSV файлами)

## Шаги

### 1. Подготовка данных на локальной машине (Windows с MT5)

На вашей локальной машине с MT5:

```bash
# Подготовить данные (например, на 12 месяцев)
python prepare_gold_data.py --months 12 --no-ask

# Это создаст файлы:
# - data/gold_train.csv
# - data/gold_val.csv  
# - data/gold_test.csv
# - data/gold_data_12months.csv
```

### 2. Упаковка данных для отправки

Используйте скрипт `upload_to_paperspace.py`:

```bash
# Создать архив со всеми данными (включая тики, если нужны)
python upload_to_paperspace.py --create-archive

# Или без тиков (если они слишком большие)
python upload_to_paperspace.py --create-archive --no-ticks

# Указать имя файла
python upload_to_paperspace.py --create-archive --output my_data.tar.gz
```

Скрипт автоматически:
- Проверит наличие всех необходимых файлов
- Покажет размеры данных
- Спросит о включении тиков (если они большие)
- Создаст сжатый tar.gz архив

### 3. Загрузка на Paperspace

**Вариант A: Через Paperspace UI (рекомендуется)**

1. Войдите в Paperspace Console
2. Перейдите в Storage
3. Загрузите созданный `.tar.gz` архив
4. Подключите storage к вашей машине

**Вариант B: Через SCP/RSYNC**

```bash
# Если настроен SSH доступ
python upload_to_paperspace.py --create-archive --upload-scp \
    --host your-paperspace-host \
    --path /storage/ \
    --user paperspace
```

### 4. Настройка на Paperspace

После загрузки архива на Paperspace:

```bash
# 1. Клонировать репозиторий
git clone https://github.com/5nail000/xauusd-ai-ea.git
cd xauusd-ai-ea

# 2. Распаковать данные
tar -xzf data_for_paperspace_*.tar.gz

# 3. Установить зависимости (БЕЗ MetaTrader5)
pip install -r requirements_linux.txt

# 4. Проверить наличие данных
ls -lh data/gold_*.csv
```

### 5. Запуск обучения

```bash
# Обучение одной модели
python train_model.py

# Или обучение всех моделей
python train_all_models.py --months 12 --batch-size 32 --epochs 100
```

## Оптимизация памяти

### Исправление float32 (уже применено)

Код автоматически использует `float32` вместо `float64` для последовательностей, что уменьшает использование памяти в 2 раза:

- **Было**: ~30 ГБ для 155k образцов с 430 фичами
- **Стало**: ~15 ГБ (в пределах 30 ГБ на Paperspace)

### Дополнительные оптимизации (если нужно)

Если все еще не хватает памяти:

1. **Уменьшить batch_size**:
   ```bash
   python train_all_models.py --batch-size 16  # вместо 32
   ```

2. **Уменьшить количество месяцев**:
   ```bash
   python prepare_gold_data.py --months 6  # вместо 12
   ```

3. **Удалить коррелированные фичи**:
   В `prepare_gold_data.py` можно включить удаление коррелированных фичей

## Структура данных

Архив содержит:

```
data/
├── gold_train.csv          # Обучающая выборка (обязательно)
├── gold_val.csv            # Валидационная выборка (обязательно)
├── gold_test.csv           # Тестовая выборка (обязательно)
├── prepared/               # Подготовленные данные (опционально)
│   └── *.pkl
└── ticks/                  # Тиковые данные (опционально, может быть большим)
    └── XAUUSD_*.pkl
```

## Требования

### На локальной машине (Windows):
- Python 3.8+
- MetaTrader5 (для подготовки данных)
- Все зависимости из `requirements.txt`

### На Paperspace (Linux):
- Python 3.8+
- Все зависимости из `requirements_linux.txt` (БЕЗ MetaTrader5)
- 30+ ГБ RAM (рекомендуется)

## Устранение проблем

### Ошибка: "Unable to allocate memory"

**Решение**: Исправление float32 уже применено. Если все еще не хватает:
- Уменьшите batch_size
- Уменьшите количество месяцев данных
- Используйте машину с большим RAM

### Ошибка: "MetaTrader5 not found"

**Решение**: Это нормально на Paperspace! MT5 нужен только для подготовки данных на Windows. Для обучения он не требуется.

### Ошибка: "File not found: data/gold_train.csv"

**Решение**: Убедитесь, что:
1. Архив распакован в правильную директорию
2. Вы находитесь в корне проекта
3. CSV файлы были созданы на локальной машине

## Следующие шаги

После успешного обучения на Paperspace:

1. Скачайте обученную модель (`models/checkpoints/*.pth`)
2. Скачайте scaler (`models/feature_scaler.pkl`)
3. Используйте их на локальной машине для бэктестинга и торговли

## Полезные команды

```bash
# Проверить размер данных
du -sh data/

# Проверить использование памяти во время обучения
watch -n 1 free -h

# Мониторинг GPU (если используется)
nvidia-smi
```

