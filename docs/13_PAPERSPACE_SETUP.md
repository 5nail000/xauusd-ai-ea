# Настройка обучения на Paperspace

## Обзор

Это руководство поможет вам перенести подготовленные данные на Paperspace и запустить обучение модели там, где доступно больше ресурсов (30 ГБ RAM).

## Преимущества Paperspace

- ✅ 30 ГБ оперативной памяти (достаточно для обучения на 12+ месяцах данных)
- ✅ GPU ускорение (опционально)
- ✅ Не требует MT5 (обучение работает только с CSV файлами)
- ✅ Поддержка offline режима для подготовки данных без MT5

## Шаги

### 1. Подготовка данных

**Вариант A: На локальной машине с MT5 (Windows)**

На вашей локальной машине с MT5:

```bash
# Подготовить данные (например, на 12 месяцев)
python prepare_gold_data.py --months 12 --no-ask

# Это создаст файлы:
# - workspace/prepared/features/gold_train.csv
# - workspace/prepared/features/gold_val.csv  
# - workspace/prepared/features/gold_test.csv
# - workspace/prepared/features/gold_data_12months.csv
# - workspace/raw_data/ticks/ (кэшированные тики)
```

**Вариант B: На Paperspace в offline режиме (Linux)**

Если у вас уже есть кэшированные тики, можно подготовить данные прямо на Paperspace:

```bash
# Подготовка данных в offline режиме (без MT5)
python prepare_gold_data.py --offline --days 30 --no-ask

# Или полный цикл
python full_pipeline.py --offline --days 30
```

**Требования для offline режима:**
- Тики должны быть загружены в `workspace/raw_data/ticks/`
- Данные должны покрывать требуемый период

### 2. Упаковка данных для отправки

Используйте скрипт `paperspace_utils.py`:

```bash
# Создать архив со всеми данными (включая тики, если нужны)
python paperspace_utils.py create-training-archive --include-ticks

# Или без тиков (если они слишком большие)
python paperspace_utils.py create-training-archive

# Указать имя файла
python paperspace_utils.py create-training-archive --output my_data.tar.gz

# Включить кэши
python paperspace_utils.py create-training-archive --include-cache
```

Скрипт автоматически:
- Проверит наличие всех необходимых файлов
- Покажет размеры данных
- Спросит о включении тиков (если они большие и не указан `--no-ask-ticks`)
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
python paperspace_utils.py upload-training \
    --host your-paperspace-host \
    --path /storage/ \
    --user paperspace \
    --include-ticks
```

**Вариант C: Через Hugging Face Hub (рекомендуется для тиков)**

Для тиковых данных удобнее использовать Hugging Face Hub:

```bash
# Загрузить тики на Hugging Face
python paperspace_utils.py hf-upload-ticks --repo-id username/xauusd-ticks

# Загрузить данные для обучения (без тиков) на Hugging Face
python paperspace_utils.py hf-upload-training --repo-id username/xauusd-training-data
```

**Преимущества Hugging Face:**
- ✅ Бесплатное хранилище
- ✅ Версионирование данных
- ✅ Легкий доступ с любой машины
- ✅ Не требует настройки SSH

**Настройка API-ключа Hugging Face:**

Для работы с Hugging Face Hub необходим токен доступа. Есть два способа его указать:

**Способ 1: Переменная окружения (рекомендуется)**

```bash
# Windows (PowerShell)
$env:HF_TOKEN="your_huggingface_token_here"

# Windows (CMD)
set HF_TOKEN=your_huggingface_token_here

# Linux/Mac
export HF_TOKEN="your_huggingface_token_here"

# Или добавить в ~/.bashrc или ~/.zshrc для постоянного использования
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
source ~/.bashrc
```

**Способ 2: Параметр командной строки**

```bash
python paperspace_utils.py hf-upload-ticks --repo-id username/xauusd-ticks --token your_huggingface_token_here
```

**Как получить токен Hugging Face:**

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Перейдите в [Settings → Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте новый токен (тип: "Write" для загрузки, "Read" для скачивания)
4. Скопируйте токен и сохраните его в переменную окружения `HF_TOKEN`

**Примечание:** Если токен не указан, система попытается использовать авторизацию через браузер (если доступен).

### 4. Настройка на Paperspace

После загрузки данных на Paperspace:

```bash
# 1. Клонировать репозиторий
git clone https://github.com/5nail000/xauusd-ai-ea.git
cd xauusd-ai-ea

# 2. Установить зависимости (БЕЗ MetaTrader5)
pip install -r requirements_linux.txt

# 3. Загрузить данные (выберите один из вариантов)
```

**Вариант A: Распаковать архив (если загружали через SCP/UI)**
```bash
tar -xzf training_data_*.tar.gz
```

**Вариант B: Скачать с Hugging Face (рекомендуется)**
```bash
# Убедитесь, что HF_TOKEN установлен (см. раздел "Настройка API-ключа" выше)
export HF_TOKEN="your_huggingface_token_here"

# Скачать тики (если нужны для offline режима)
python paperspace_utils.py hf-download-ticks --repo-id username/xauusd-ticks

# Скачать данные для обучения
python paperspace_utils.py hf-download-training --repo-id username/xauusd-training-data

# Или указать токен напрямую
python paperspace_utils.py hf-download-ticks --repo-id username/xauusd-ticks --token your_huggingface_token_here
```

**4. Проверить наличие данных**
```bash
ls -lh workspace/prepared/features/gold_*.csv
```

### 5. Запуск обучения

```bash
# Обучение одной модели
python train_model.py

# Или обучение всех моделей
python train_all_models.py --months 12 --batch-size 32 --epochs 100
```

### 6. Подготовка данных на Paperspace (offline режим)

Если у вас уже есть кэшированные тики, можно подготовить данные прямо на Paperspace без MT5:

```bash
# Подготовка данных в offline режиме
python prepare_gold_data.py --offline --days 30

# Полный цикл: подготовка + обучение + бэктестинг
python full_pipeline.py --offline --days 30

# С удалением коррелированных фичей
python full_pipeline.py --offline --days 30 --remove-correlated
```

**Преимущества offline режима на Paperspace:**
- ✅ Не нужно загружать подготовленные CSV файлы
- ✅ Можно подготовить данные прямо на Paperspace
- ✅ Старшие таймфреймы создаются автоматически из минутных данных
- ✅ Полная совместимость с обычным режимом

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

Архив или Hugging Face репозиторий содержит:

```
workspace/
├── prepared/
│   ├── features/
│   │   ├── gold_train.csv          # Обучающая выборка (обязательно)
│   │   ├── gold_val.csv            # Валидационная выборка (обязательно)
│   │   └── gold_test.csv           # Тестовая выборка (обязательно)
│   └── scalers/                    # Scalers (опционально)
│       └── feature_scaler_*.pkl
├── raw_data/
│   ├── cache/                      # Кэши (опционально)
│   │   └── *.pkl
│   └── ticks/                      # Тиковые данные (опционально, может быть большим)
│       ├── XAUUSD_*.parquet        # Тики в формате Parquet (кроссплатформенный)
│       └── XAUUSD_*.pkl            # Старые тики в формате Pickle (автоматически конвертируются)
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

### Ошибка: "HF_TOKEN не установлен" или "Authentication required"

**Решение**: Установите токен Hugging Face одним из способов:

1. **Через переменную окружения (рекомендуется):**
   ```bash
   # Linux/Mac
   export HF_TOKEN="your_huggingface_token_here"
   
   # Windows PowerShell
   $env:HF_TOKEN="your_huggingface_token_here"
   
   # Windows CMD
   set HF_TOKEN=your_huggingface_token_here
   ```

2. **Через параметр командной строки:**
   ```bash
   python paperspace_utils.py hf-upload-ticks --repo-id username/dataset --token your_token
   ```

3. **Получите токен на [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)**
   - Создайте новый токен (тип: "Write" для загрузки, "Read" для скачивания)
   - Скопируйте токен и сохраните его

**Примечание:** Если токен не указан, система попытается использовать авторизацию через браузер (если доступен).

### Ошибка: "Unable to allocate memory"

**Решение**: Исправление float32 уже применено. Если все еще не хватает:
- Уменьшите batch_size
- Уменьшите количество месяцев данных
- Используйте машину с большим RAM

### Ошибка: "MetaTrader5 not found"

**Решение**: Это нормально на Paperspace! MT5 нужен только для подготовки данных на Windows. Для обучения он не требуется.

### Ошибка: "File not found: workspace/prepared/features/gold_train.csv"

**Решение**: Убедитесь, что:
1. Архив распакован в правильную директорию
2. Вы находитесь в корне проекта
3. CSV файлы были созданы на локальной машине

## Следующие шаги

После успешного обучения на Paperspace:

1. Скачайте обученную модель (`workspace/models/checkpoints/*.pth`)
2. Скачайте scaler (`workspace/prepared/scalers/feature_scaler.pkl`)
3. Используйте их на локальной машине для бэктестинга и торговли

**Или используйте `paperspace_utils.py` для автоматизации:**

```bash
# Создать архив с результатами
python paperspace_utils.py create-results-archive

# Скачать результаты через SCP
python paperspace_utils.py download-results --host paperspace.com --path /storage/results.tar.gz

# Или загрузить результаты на Hugging Face
python paperspace_utils.py hf-upload-training --repo-id username/xauusd-results \
    --include-scalers
```

## Полезные команды

```bash
# Проверить размер данных
du -sh data/

# Проверить использование памяти во время обучения
watch -n 1 free -h

# Мониторинг GPU (если используется)
nvidia-smi
```

