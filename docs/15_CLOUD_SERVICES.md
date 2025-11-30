# Руководство по облачным сервисам

## Обзор

Модуль `cloud_services.py` предоставляет утилиты для работы с облачными сервисами:
- **Paperspace** - для обучения моделей на мощных серверах
- **Hugging Face Hub** - для хранения, обмена и управления данными, моделями и результатами анализа
  - Загрузка данных (тики, данные для обучения, результаты анализа)
  - Скачивание данных
  - Удаление данных (отдельные группы или все данные)

## Установка зависимостей

```bash
# Для работы с Hugging Face
pip install huggingface_hub>=0.20.0

# Для работы с Paperspace API (обязательно для API команд)
pip install paperspace>=0.2.0

# Для работы с Paperspace (SCP/RSYNC)
# Windows: установите OpenSSH или Git Bash
# Linux/Mac: уже включены в систему
```

## Настройка

### Hugging Face

1. Создайте аккаунт на [Hugging Face](https://huggingface.co/)
2. Создайте токен доступа: Settings → Access Tokens → New Token
3. Установите токен одним из способов:

```bash
# Вариант 1: Переменная окружения (рекомендуется)
export HF_TOKEN=your_token_here  # Linux/Mac
set HF_TOKEN=your_token_here      # Windows CMD
$env:HF_TOKEN="your_token_here"  # Windows PowerShell

# Вариант 2: Параметр --token в командах
python cloud_services.py hf-upload-ticks --repo-id username/dataset --token your_token
```

### Paperspace

**Для работы через SSH (SCP/RSYNC):**
1. Настройте SSH доступ к вашему Paperspace серверу
2. Убедитесь, что у вас есть доступ к `/storage/` или другой директории

**Для работы через Paperspace API (автоматизация):**
1. **Установите paperspace CLI:**
   ```bash
   pip install paperspace
   ```

2. **Проверьте установку:**
   ```bash
   python -m paperspace version
   ```

3. **Получите API ключ:**
   - Зайдите в [Paperspace Console](https://console.paperspace.com/)
   - Перейдите в Settings → API Keys
   - Создайте новый API ключ (не Core API - он устарел с 15 июля 2024)

4. **Установите переменную окружения:**
   ```bash
   # Linux/Mac
   export PAPERSPACE_API_KEY=your_api_key_here

   # Windows PowerShell
   $env:PAPERSPACE_API_KEY="your_api_key_here"

   # Windows CMD
   set PAPERSPACE_API_KEY=your_api_key_here
   ```

5. Или передавайте ключ через параметр `--api-key` в командах

**⚠️ Важно:** Core API устарел с 15 июля 2024. Используйте paperspace CLI.

---

## Paperspace: Работа с данными для обучения

### 1. Создание архива с данными

Создает tar.gz архив с подготовленными данными для обучения.

```bash
# Базовый вариант (только CSV файлы)
python cloud_services.py create-training-archive

# С указанием имени файла
python cloud_services.py create-training-archive --output my_training_data.tar.gz

# Включить тиковые данные (с подтверждением)
python cloud_services.py create-training-archive --include-ticks

# Включить тиковые данные (без подтверждения)
python cloud_services.py create-training-archive --include-ticks --no-ask-ticks

# Включить кэши
python cloud_services.py create-training-archive --include-cache

# Все опции вместе
python cloud_services.py create-training-archive \
  --output training_full.tar.gz \
  --include-ticks \
  --include-cache \
  --no-ask-ticks
```

**Что включается в архив:**
- `workspace/prepared/features/gold_train.csv` (обязательно)
- `workspace/prepared/features/gold_val.csv` (обязательно)
- `workspace/prepared/features/gold_test.csv` (обязательно)
- `workspace/raw_data/ticks/` (опционально, если `--include-ticks`)
- `workspace/raw_data/cache/` (опционально, если `--include-cache`)

**Параметры:**
- `--output, -o` - путь к выходному архиву (по умолчанию: `training_data_YYYYMMDD_HHMMSS.tar.gz`)
- `--include-ticks` - включить тиковые данные
- `--include-cache` - включить кэши
- `--no-ask-ticks` - не спрашивать подтверждение для тиков

### 2. Загрузка данных на Paperspace

Загружает архив с данными на Paperspace сервер.

```bash
# Базовый вариант (создает архив и загружает)
python cloud_services.py upload-training \
  --host your-paperspace-host \
  --path /storage/

# С указанием пользователя
python cloud_services.py upload-training \
  --host your-paperspace-host \
  --path /storage/ \
  --user paperspace

# Использовать rsync вместо scp
python cloud_services.py upload-training \
  --host your-paperspace-host \
  --path /storage/ \
  --method rsync

# С тиковыми данными
python cloud_services.py upload-training \
  --host your-paperspace-host \
  --path /storage/ \
  --include-ticks
```

**Параметры:**
- `--host` - хост Paperspace (по умолчанию: `paperspace.com`)
- `--path` - путь на сервере (по умолчанию: `/storage/`)
- `--user` - имя пользователя для SSH (опционально)
- `--method` - метод загрузки: `scp` или `rsync` (по умолчанию: `scp`)
- `--include-ticks` - включить тиковые данные в архив
- `--include-cache` - включить кэши в архив
- `--no-ask-ticks` - не спрашивать подтверждение для тиков

**После загрузки на Paperspace:**

```bash
# Подключитесь к серверу
ssh paperspace@your-paperspace-host

# Распакуйте архив
cd /storage/
tar -xzf training_data_*.tar.gz

# Теперь можно запускать обучение
python full_pipeline.py --months 12 --skip-prepare
```

### 3. Просмотр файлов на Paperspace

```bash
python cloud_services.py list-remote-files \
  --host your-paperspace-host \
  --path /storage/
```

---

## Paperspace: Работа с результатами обучения

### 1. Создание архива с результатами

Создает tar.gz архив с результатами обучения (модели, метрики, scalers).

```bash
# Базовый вариант
python cloud_services.py create-results-archive

# С указанием имени файла
python cloud_services.py create-results-archive --output results_20240101.tar.gz
```

**Что включается в архив:**
- `workspace/models/checkpoints/` - обученные модели
- `workspace/models/metrics/` - метрики обучения
- `workspace/prepared/scalers/` - scalers для нормализации
- `workspace/models/logs/` - логи TensorBoard (если есть)

### 2. Скачивание результатов с Paperspace

```bash
# Скачать архив с результатами
python cloud_services.py download-results \
  --host your-paperspace-host \
  --path /storage/results_20240101.tar.gz \
  --local-path ./

# Использовать rsync
python cloud_services.py download-results \
  --host your-paperspace-host \
  --path /storage/results_20240101.tar.gz \
  --method rsync \
  --local-path ./results/
```

**Параметры:**
- `--host` - хост Paperspace
- `--path` - путь к архиву на сервере (обязательно)
- `--user` - имя пользователя для SSH (опционально)
- `--method` - метод скачивания: `scp` или `rsync` (по умолчанию: `scp`)
- `--local-path` - локальная директория для сохранения (по умолчанию: `.`)

**После скачивания:**

```bash
# Распакуйте архив
tar -xzf results_20240101.tar.gz

# Структура будет восстановлена:
# workspace/models/checkpoints/
# workspace/models/metrics/
# workspace/prepared/scalers/
```

---

## Paperspace: Автоматизация через API

Paperspace CLI позволяет полностью автоматизировать процесс обучения: создание машин, клонирование репозитория, установку зависимостей, запуск обучения и загрузку результатов на Hugging Face.

### Преимущества API подхода

- ✅ **Полная автоматизация** - весь процесс от создания машины до загрузки модели
- ✅ **Не требует SSH** - все через API запросы
- ✅ **Управление жизненным циклом** - автоматическое создание и остановка машин
- ✅ **Интеграция с Hugging Face** - автоматическая загрузка моделей

### 1. Создание машины через API

**⚠️ Важно:** 
- Core API устарел с 15 июля 2024 и больше не доступен
- Используется paperspace CLI через subprocess
- Убедитесь, что paperspace установлен: `pip install paperspace`
- Проверьте доступность: `python -m paperspace version`

```bash
# Базовый вариант
python cloud_services.py ps-create-machine

# С указанием типа машины
python cloud_services.py ps-create-machine \
  --machine-type P5000 \
  --template PyTorch \
  --region NYC1 \
  --size 100 \
  --billing-type hourly

# С кастомным именем и месячной оплатой
python cloud_services.py ps-create-machine \
  --machine-type GPU+ \
  --name my-training-machine \
  --size 50 \
  --billing-type monthly \
  --api-key your_paperspace_api_key
```

**Если возникает ошибка:**
- Проверьте, что paperspace CLI установлен: `pip install paperspace`
- Проверьте доступность: `python -m paperspace version`
- Убедитесь, что используете правильный API ключ
- Проверьте параметры команды: `python -m paperspace machines create --help`

**Параметры:**
- `--machine-type` - тип машины: `GPU+`, `P4000`, `P5000`, etc. (по умолчанию: `GPU+`)
- `--template` - ID шаблона образа (по умолчанию: `PyTorch`)
- `--region` - регион: `NYC1` → `ny2`, `AMS1` → `ams1`, `CA1` → `ca1` (по умолчанию: `NYC1`)
- `--name` - имя машины (опционально, генерируется автоматически)
- `--size` - размер хранилища в GB (обязательно, по умолчанию: `50`)
- `--billing-type` - тип оплаты: `hourly` или `monthly` (по умолчанию: `hourly`)
- `--api-key` - Paperspace API ключ (или используйте `PAPERSPACE_API_KEY` env var)

**Результат:**
Команда выведет ID созданной машины, которую можно использовать для дальнейших операций.

### 2. Полный цикл обучения через API

**Самая мощная функция** - автоматизирует весь процесс:

1. Создает машину на Paperspace
2. Клонирует репозиторий
3. Устанавливает зависимости из `requirements_linux.txt`
4. Скачивает датасет с Hugging Face
5. Запускает обучение
6. Загружает модель на Hugging Face
7. Останавливает машину

```bash
# Полный цикл обучения
python cloud_services.py ps-run-training \
  --repo-url https://github.com/username/xauusd-ai-ea.git \
  --hf-dataset-repo username/xauusd-training-data \
  --hf-model-repo username/xauusd-models \
  --months 12

# С указанием типа машины и токенов
python cloud_services.py ps-run-training \
  --repo-url https://github.com/username/xauusd-ai-ea.git \
  --hf-dataset-repo username/xauusd-training-data \
  --hf-model-repo username/xauusd-models \
  --machine-type P5000 \
  --months 12 \
  --hf-token your_hf_token \
  --api-key your_paperspace_api_key
```

**Параметры:**
- `--repo-url` - URL репозитория для клонирования (обязательно)
- `--hf-dataset-repo` - ID репозитория датасета на Hugging Face (обязательно)
- `--hf-model-repo` - ID репозитория модели на Hugging Face (обязательно)
- `--machine-type` - тип машины (по умолчанию: `GPU+`)
- `--months` - количество месяцев данных (по умолчанию: `12`)
- `--hf-token` - Hugging Face токен (или используйте `HF_TOKEN` env var)
- `--api-key` - Paperspace API ключ (или используйте `PAPERSPACE_API_KEY` env var)

**Что происходит внутри:**

1. **Создание машины** - создается машина указанного типа
2. **Ожидание готовности** - система ждет, пока машина станет готовой
3. **Настройка окружения** - выполняется скрипт:
   ```bash
   cd /storage
   git clone <repo-url> xauusd-ai-ea
   cd xauusd-ai-ea
   pip install -r requirements_linux.txt
   ```
4. **Скачивание датасета** - автоматически скачиваются данные с Hugging Face
5. **Запуск обучения** - выполняется `full_pipeline.py --skip-prepare --months <N> --skip-backtest`
6. **Загрузка модели** - обученная модель загружается на Hugging Face
7. **Остановка машины** - машина автоматически останавливается

**Пример использования на Windows Server:**

```powershell
# Установить переменные окружения
$env:PAPERSPACE_API_KEY="your_paperspace_api_key"
$env:HF_TOKEN="your_huggingface_token"

# Запустить полный цикл обучения
python cloud_services.py ps-run-training `
  --repo-url https://github.com/username/xauusd-ai-ea.git `
  --hf-dataset-repo username/xauusd-training-data `
  --hf-model-repo username/xauusd-models `
  --months 12
```

### 3. Управление машинами

```bash
# Остановить машину
python cloud_services.py ps-stop-machine \
  --machine-id ps123abc

# Удалить машину
python cloud_services.py ps-delete-machine \
  --machine-id ps123abc
```

**Параметры:**
- `--machine-id` - ID машины (обязательно)
- `--api-key` - Paperspace API ключ (опционально)

### Программируемый API

Вы также можете использовать API программно:

```python
from cloud_services import PaperspaceAPIClient

# Создать клиент
client = PaperspaceAPIClient(api_key="your_api_key")

# Создать машину
machine = client.create_machine(
    machine_type="GPU+",
    template="PyTorch",
    region="NYC1",
    name="my-training-machine"
)

# Ожидать готовности
client.wait_for_machine_ready(machine['id'])

# Запустить полный цикл обучения
success = client.run_full_training_pipeline(
    repo_url="https://github.com/username/xauusd-ai-ea.git",
    hf_dataset_repo="username/xauusd-training-data",
    hf_model_repo="username/xauusd-models",
    hf_token="your_hf_token",
    machine_type="GPU+",
    months=12
)

# Остановить машину
client.stop_machine(machine['id'])
```

### Типичный сценарий: Автоматическое обучение

```bash
# На Windows Server или любой машине с Python

# 1. Установить переменные окружения
export PAPERSPACE_API_KEY="your_paperspace_api_key"
export HF_TOKEN="your_huggingface_token"

# 2. Запустить полный цикл
python cloud_services.py ps-run-training \
  --repo-url https://github.com/username/xauusd-ai-ea.git \
  --hf-dataset-repo username/xauusd-training-data \
  --hf-model-repo username/xauusd-models \
  --months 12

# 3. Готово! Модель загружена на Hugging Face
```

**Преимущества:**
- Не нужно вручную подключаться к Paperspace
- Не нужно настраивать SSH
- Весь процесс автоматизирован
- Машина автоматически останавливается после завершения

---

## Hugging Face: Работа с тиковыми данными

### 1. Загрузка тиков на Hugging Face

Загружает тиковые данные из `workspace/raw_data/ticks/` на Hugging Face Hub.

```bash
# Базовый вариант
python cloud_services.py hf-upload-ticks \
  --repo-id username/xauusd-ticks

# С указанием токена
python cloud_services.py hf-upload-ticks \
  --repo-id username/xauusd-ticks \
  --token your_huggingface_token

# С указанием директории
python cloud_services.py hf-upload-ticks \
  --repo-id username/xauusd-ticks \
  --ticks-dir workspace/raw_data/ticks

# С кастомным сообщением коммита
python cloud_services.py hf-upload-ticks \
  --repo-id username/xauusd-ticks \
  --commit-message "Upload tick data for 2024"
```

**Параметры:**
- `--repo-id` - ID репозитория в формате `username/dataset-name` (обязательно)
- `--token` - Hugging Face токен (если не установлен `HF_TOKEN`)
- `--ticks-dir` - директория с тиками (по умолчанию: `workspace/raw_data/ticks`)
- `--commit-message` - сообщение коммита (по умолчанию: "Upload tick data")

**Перед загрузкой:**
1. Создайте репозиторий на [Hugging Face](https://huggingface.co/new-dataset)
2. Выберите тип: **Dataset**
3. Укажите имя репозитория (например: `xauusd-ticks`)

### 2. Скачивание тиков с Hugging Face

```bash
# Базовый вариант
python cloud_services.py hf-download-ticks \
  --repo-id username/xauusd-ticks

# С указанием локальной директории
python cloud_services.py hf-download-ticks \
  --repo-id username/xauusd-ticks \
  --local-dir workspace/raw_data/ticks

# С указанием токена (для приватных репозиториев)
python cloud_services.py hf-download-ticks \
  --repo-id username/xauusd-ticks \
  --token your_huggingface_token
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен (если не установлен `HF_TOKEN`)
- `--local-dir` - локальная директория для сохранения (по умолчанию: `workspace/raw_data/ticks`)

**Использование скачанных тиков:**

После скачивания тики будут доступны для offline режима:

```bash
# Подготовка данных в offline режиме
python prepare_gold_data.py --offline --months 6
```

---

## Hugging Face: Работа с данными для обучения

### 1. Загрузка данных для обучения

Загружает CSV файлы и scalers на Hugging Face Hub.

```bash
# Базовый вариант (с scalers)
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-training-data

# Без scalers
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-training-data \
  --no-scalers

# С кэшами
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-training-data \
  --include-cache

# С кастомным сообщением
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-training-data \
  --commit-message "Upload training data v2.0"
```

**Что включается:**
- `workspace/prepared/features/gold_train.csv`
- `workspace/prepared/features/gold_val.csv`
- `workspace/prepared/features/gold_test.csv`
- `workspace/prepared/scalers/` (если `--include-scalers`, по умолчанию включено)
- `workspace/raw_data/cache/` (если `--include-cache`)

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--include-scalers` - включить scalers (по умолчанию: включено)
- `--no-scalers` - не включать scalers
- `--include-cache` - включить кэши
- `--commit-message` - сообщение коммита (по умолчанию: "Upload training data")

### 2. Скачивание данных для обучения

```bash
# Базовый вариант
python cloud_services.py hf-download-training \
  --repo-id username/xauusd-training-data

# С указанием локальной директории
python cloud_services.py hf-download-training \
  --repo-id username/xauusd-training-data \
  --local-dir workspace
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--local-dir` - локальная директория (по умолчанию: `workspace`)

**После скачивания:**

Структура будет восстановлена:
```
workspace/
├── prepared/
│   ├── features/
│   │   ├── gold_train.csv
│   │   ├── gold_val.csv
│   │   └── gold_test.csv
│   └── scalers/  (если были включены)
└── raw_data/
    └── cache/  (если были включены)
```

Теперь можно запускать обучение:

```bash
python full_pipeline.py --skip-prepare
```

---

## Hugging Face: Работа с результатами анализа фичей

### 1. Загрузка результатов анализа фичей

Загружает результаты анализа фичей (из `--analyze-features`) на Hugging Face Hub.

**Сначала запустите анализ:**

```bash
# Запустить анализ фичей
python full_pipeline.py --months 6 --analyze-features

# Или отдельно
python analyze_and_exclude_features.py \
  --train workspace/prepared/features/gold_train.csv \
  --val workspace/prepared/features/gold_val.csv \
  --test workspace/prepared/features/gold_test.csv
```

**Затем загрузите результаты:**

```bash
# Базовый вариант
python cloud_services.py hf-upload-features \
  --repo-id username/xauusd-feature-analysis

# С указанием директории
python cloud_services.py hf-upload-features \
  --repo-id username/xauusd-feature-analysis \
  --analysis-dir workspace/analysis-of-features

# С кастомным сообщением
python cloud_services.py hf-upload-features \
  --repo-id username/xauusd-feature-analysis \
  --commit-message "Feature analysis for 6 months data"
```

**Что включается:**
- `feature_statistics.csv` - базовая статистика по фичам
- `feature_importance.csv` - важность фичей (Mutual Info, F-score)
- `outliers_analysis.csv` - анализ выбросов
- `feature_by_class_statistics.csv` - статистика по классам
- `feature_analysis_report.html` - сводный HTML отчет
- `plots/` - графики (если были сгенерированы)
  - `distributions/` - распределения фичей
  - `by_class/` - распределения по классам

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--analysis-dir` - директория с результатами (по умолчанию: `workspace/analysis-of-features`)
- `--commit-message` - сообщение коммита (по умолчанию: "Upload feature analysis results")

### 2. Скачивание результатов анализа фичей

```bash
# Базовый вариант
python cloud_services.py hf-download-features \
  --repo-id username/xauusd-feature-analysis

# С указанием локальной директории
python cloud_services.py hf-download-features \
  --repo-id username/xauusd-feature-analysis \
  --local-dir workspace/analysis-of-features
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--local-dir` - локальная директория (по умолчанию: `workspace/analysis-of-features`)

**После скачивания:**

Все файлы будут доступны в `workspace/analysis-of-features/`:
- CSV файлы с анализом
- HTML отчет для просмотра в браузере
- Графики (если были загружены)

---

## Hugging Face: Удаление данных

### 1. Удаление тиков

Удаляет тиковые данные из репозитория (папка `ticks/`).

```bash
# Базовый вариант
python cloud_services.py hf-delete-ticks \
  --repo-id username/xauusd-ticks

# С кастомным сообщением коммита
python cloud_services.py hf-delete-ticks \
  --repo-id username/xauusd-ticks \
  --commit-message "Remove old tick data"
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--commit-message` - сообщение коммита (по умолчанию: "Delete tick data")

**Что удаляется:**
- Все файлы и директории в папке `ticks/`

### 2. Удаление данных для обучения

Удаляет данные для обучения из репозитория.

```bash
# Удалить все данные для обучения (CSV, scalers, кэши)
python cloud_services.py hf-delete-training \
  --repo-id username/xauusd-training-data

# Удалить только CSV файлы (без scalers)
python cloud_services.py hf-delete-training \
  --repo-id username/xauusd-training-data \
  --no-scalers

# Удалить только CSV файлы (без кэшей)
python cloud_services.py hf-delete-training \
  --repo-id username/xauusd-training-data \
  --no-cache

# Удалить только CSV файлы (без scalers и кэшей)
python cloud_services.py hf-delete-training \
  --repo-id username/xauusd-training-data \
  --no-scalers \
  --no-cache
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--no-scalers` - не удалять scalers (по умолчанию удаляются)
- `--no-cache` - не удалять кэши (по умолчанию удаляются)
- `--commit-message` - сообщение коммита (по умолчанию: "Delete training data")

**Что удаляется:**
- `workspace/prepared/features/gold_train.csv`
- `workspace/prepared/features/gold_val.csv`
- `workspace/prepared/features/gold_test.csv`
- `workspace/prepared/scalers/` (если не указан `--no-scalers`)
- `workspace/raw_data/cache/` (если не указан `--no-cache`)
- `workspace/excluded_features.txt`

### 3. Удаление результатов анализа фичей

Удаляет результаты анализа фичей из репозитория (папка `analysis-of-features/`).

```bash
# Базовый вариант
python cloud_services.py hf-delete-features \
  --repo-id username/xauusd-feature-analysis

# С кастомным сообщением коммита
python cloud_services.py hf-delete-features \
  --repo-id username/xauusd-feature-analysis \
  --commit-message "Remove old analysis results"
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--commit-message` - сообщение коммита (по умолчанию: "Delete feature analysis results")

**Что удаляется:**
- Все файлы и директории в папке `analysis-of-features/`

### 4. Удаление всех данных (очистка датасета)

**⚠️ ВНИМАНИЕ:** Эта команда удаляет ВСЕ данные из репозитория. Используйте с осторожностью!

Удаляет все данные из датасета, очищая репозиторий для новых загрузок.

```bash
# Удалить все данные (требует подтверждения)
python cloud_services.py hf-delete-all \
  --repo-id username/xauusd-dataset

# С кастомным сообщением коммита
python cloud_services.py hf-delete-all \
  --repo-id username/xauusd-dataset \
  --commit-message "Clear dataset for new uploads"
```

**Параметры:**
- `--repo-id` - ID репозитория (обязательно)
- `--token` - Hugging Face токен
- `--commit-message` - сообщение коммита (по умолчанию: "Delete all dataset data")

**Безопасность:**
- Команда требует подтверждения: нужно ввести `yes` для продолжения
- Показывает список всех файлов перед удалением
- Показывает прогресс удаления

**Что удаляется:**
- Все файлы и директории в репозитории
- Репозиторий остается пустым и готовым для новых загрузок

**Пример использования:**

```bash
# 1. Очистить датасет от старых данных
python cloud_services.py hf-delete-all \
  --repo-id username/xauusd-dataset
# Введите: yes

# 2. Загрузить новые данные
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-dataset
```

---

## Типичные сценарии использования

### Сценарий 1: Обучение на Paperspace

```bash
# 1. На локальной машине: подготовить данные
python prepare_gold_data.py --months 12

# 2. Создать и загрузить архив на Paperspace
python cloud_services.py upload-training \
  --host your-paperspace-host \
  --path /storage/ \
  --include-ticks

# 3. На Paperspace: распаковать и обучить
ssh paperspace@your-paperspace-host
cd /storage/
tar -xzf training_data_*.tar.gz
python full_pipeline.py --skip-prepare --months 12

# 4. Создать архив с результатами
python cloud_services.py create-results-archive

# 5. На локальной машине: скачать результаты
python cloud_services.py download-results \
  --host your-paperspace-host \
  --path /storage/results_*.tar.gz
```

### Сценарий 2: Хранение данных на Hugging Face

```bash
# 1. Загрузить тики на Hugging Face (один раз)
python cloud_services.py hf-upload-ticks \
  --repo-id username/xauusd-ticks

# 2. Загрузить подготовленные данные
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-training-data

# 3. На другой машине: скачать данные
python cloud_services.py hf-download-ticks \
  --repo-id username/xauusd-ticks

python cloud_services.py hf-download-training \
  --repo-id username/xauusd-training-data

# 4. Запустить обучение
python full_pipeline.py --skip-prepare
```

### Сценарий 3: Обмен результатами анализа

```bash
# 1. Запустить анализ фичей
python full_pipeline.py --analyze-features

# 2. Загрузить результаты на Hugging Face
python cloud_services.py hf-upload-features \
  --repo-id username/xauusd-feature-analysis

# 3. Другие пользователи могут скачать и просмотреть
python cloud_services.py hf-download-features \
  --repo-id username/xauusd-feature-analysis

# 4. Открыть HTML отчет
# workspace/analysis-of-features/feature_analysis_report.html
```

### Сценарий 4: Очистка и обновление данных на Hugging Face

```bash
# 1. Удалить старые данные для обучения
python cloud_services.py hf-delete-training \
  --repo-id username/xauusd-training-data

# 2. Загрузить новые данные
python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-training-data

# Или полностью очистить датасет и загрузить все заново:
# 1. Очистить все данные (требует подтверждения)
python cloud_services.py hf-delete-all \
  --repo-id username/xauusd-dataset
# Введите: yes

# 2. Загрузить новые данные
python cloud_services.py hf-upload-ticks \
  --repo-id username/xauusd-dataset

python cloud_services.py hf-upload-training \
  --repo-id username/xauusd-dataset

python cloud_services.py hf-upload-features \
  --repo-id username/xauusd-dataset
```

### Сценарий 5: Удаление отдельных групп данных

```bash
# Удалить только тики (оставить остальные данные)
python cloud_services.py hf-delete-ticks \
  --repo-id username/xauusd-dataset

# Удалить только результаты анализа (оставить данные для обучения)
python cloud_services.py hf-delete-features \
  --repo-id username/xauusd-dataset

# Удалить данные для обучения, но оставить scalers
python cloud_services.py hf-delete-training \
  --repo-id username/xauusd-dataset \
  --no-scalers
```

---

## Устранение проблем

### Ошибка: "huggingface_hub не установлен"

```bash
pip install huggingface_hub>=0.20.0
```

### Ошибка: "HF_TOKEN не установлен"

```bash
# Установите токен
export HF_TOKEN=your_token_here  # Linux/Mac
$env:HF_TOKEN="your_token_here"  # Windows PowerShell

# Или используйте --token в команде
python cloud_services.py hf-upload-ticks --repo-id username/dataset --token your_token
```

### Ошибка: "SCP не найден" (Windows)

Установите OpenSSH или используйте Git Bash:
- Windows 10+: OpenSSH уже включен
- Или установите [Git for Windows](https://git-scm.com/download/win)

### Ошибка: "Permission denied" при загрузке на Paperspace

Проверьте:
1. Правильность хоста и пути
2. Наличие SSH ключей
3. Права доступа на сервере

### Ошибка: "Repository not found" на Hugging Face

1. Убедитесь, что репозиторий существует
2. Проверьте правильность формата `username/dataset-name`
3. Для приватных репозиториев используйте `--token`

### Большой размер данных

Для больших датасетов:
- Используйте `--no-ask-ticks` чтобы не включать тики
- Загружайте данные частями
- Используйте Git LFS для больших файлов на Hugging Face

---

## Рекомендации

### Организация репозиториев на Hugging Face

Рекомендуемая структура:
- `username/xauusd-ticks` - тиковые данные
- `username/xauusd-training-data` - данные для обучения
- `username/xauusd-feature-analysis` - результаты анализа фичей
- `username/xauusd-models` - обученные модели (опционально)

### Безопасность

- **Никогда не коммитьте токены в код**
- Используйте переменные окружения для токенов
- Для приватных данных используйте приватные репозитории на Hugging Face
- Регулярно обновляйте токены доступа

### Оптимизация размера

- Тиковые данные могут быть очень большими (десятки ГБ)
- Используйте `--include-ticks` только при необходимости
- Рассмотрите возможность сжатия данных перед загрузкой
- На Hugging Face используйте Git LFS для файлов > 100MB

---

## Программируемый API

Если вы хотите использовать классы напрямую в Python коде, используйте следующие методы:

### Paperspace

```python
from cloud_services import PaperspaceUploader, PaperspaceDownloader

# Загрузка данных
uploader = PaperspaceUploader(host='paperspace.com', path='/storage/')
uploader.create_paperspace_training_archive('training_data.tar.gz', include_ticks=True)
uploader.upload_paperspace_training_data('training_data.tar.gz', method='scp')

# Скачивание результатов
downloader = PaperspaceDownloader(host='paperspace.com', path='/storage/')
downloader.create_paperspace_results_archive('results.tar.gz')
downloader.download_paperspace_results('/storage/results.tar.gz', local_path='./')
downloader.list_paperspace_files()
```

### Hugging Face

```python
from cloud_services import HuggingFaceUploader, HuggingFaceDownloader, HuggingFaceDeleter

# Загрузка данных
uploader = HuggingFaceUploader(repo_id='username/dataset-name')
uploader.upload_hf_ticks(ticks_dir='workspace/raw_data/ticks')
uploader.upload_hf_training_data(include_scalers=True)
uploader.upload_hf_feature_analysis(analysis_dir='workspace/analysis-of-features')

# Скачивание данных
downloader = HuggingFaceDownloader(repo_id='username/dataset-name')
downloader.download_hf_ticks(local_dir='workspace/raw_data/ticks')
downloader.download_hf_training_data(local_dir='workspace')
downloader.download_hf_feature_analysis(local_dir='workspace/analysis-of-features')

# Удаление данных
deleter = HuggingFaceDeleter(repo_id='username/dataset-name')
deleter.delete_hf_ticks()  # Удалить тики
deleter.delete_hf_training_data(include_scalers=True, include_cache=True)  # Удалить данные для обучения
deleter.delete_hf_feature_analysis()  # Удалить результаты анализа
deleter.delete_all_data()  # Удалить все данные (требует подтверждения)
```

**Примечание:** Для обратной совместимости старые названия методов (без префиксов `paperspace_` и `hf_`) также доступны как алиасы, но рекомендуется использовать новые названия для ясности.

---

## Справочник всех команд и опций

### Paperspace: Работа с данными для обучения

#### `upload-training`
Создает архив и загружает данные для обучения на Paperspace сервер.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--host` | str | Нет | `paperspace.com` | Хост Paperspace |
| `--path` | str | Нет | `/storage/` | Путь на Paperspace |
| `--user` | str | Нет | `None` | Пользователь для SSH |
| `--method` | str | Нет | `scp` | Метод загрузки: `scp` или `rsync` |
| `--include-ticks` | flag | Нет | `False` | Включить тиковые данные |
| `--include-cache` | flag | Нет | `False` | Включить кэши |
| `--no-ask-ticks` | flag | Нет | `False` | Не спрашивать подтверждение для тиков |

#### `create-training-archive`
Создает tar.gz архив с данными для обучения (без загрузки).

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--output`, `-o` | str | Нет | `training_data_YYYYMMDD_HHMMSS.tar.gz` | Путь к выходному архиву |
| `--include-ticks` | flag | Нет | `False` | Включить тиковые данные |
| `--include-cache` | flag | Нет | `False` | Включить кэши |
| `--no-ask-ticks` | flag | Нет | `False` | Не спрашивать подтверждение для тиков |

#### `download-results`
Скачивает результаты обучения с Paperspace сервера.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--host` | str | Нет | `paperspace.com` | Хост Paperspace |
| `--path` | str | **Да** | - | Путь к архиву на Paperspace |
| `--user` | str | Нет | `None` | Пользователь для SSH |
| `--method` | str | Нет | `scp` | Метод скачивания: `scp` или `rsync` |
| `--local-path` | str | Нет | `.` | Локальная директория для сохранения |

#### `create-results-archive`
Создает tar.gz архив с результатами обучения.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--output`, `-o` | str | Нет | `results_YYYYMMDD_HHMMSS.tar.gz` | Путь к выходному архиву |

#### `list-remote-files`
Выводит список файлов на Paperspace сервере.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--host` | str | Нет | `paperspace.com` | Хост Paperspace |
| `--path` | str | Нет | `/storage/` | Путь на Paperspace |
| `--user` | str | Нет | `None` | Пользователь для SSH |

### Paperspace: Автоматизация через API

#### `ps-create-machine`
Создает машину на Paperspace через API.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--machine-type` | str | Нет | `GPU+` | Тип машины (GPU+, P4000, P5000, etc.) |
| `--template` | str | Нет | `PyTorch` | ID шаблона образа |
| `--region` | str | Нет | `NYC1` | Регион (NYC1 → ny2, AMS1 → ams1, CA1 → ca1) |
| `--name` | str | Нет | Авто | Имя машины |
| `--size` | int | Нет | `50` | Размер хранилища в GB |
| `--billing-type` | str | Нет | `hourly` | Тип оплаты: `hourly` или `monthly` |
| `--api-key` | str | Нет | `PAPERSPACE_API_KEY` | Paperspace API ключ |

#### `ps-run-training`
Запускает полный цикл обучения на Paperspace через API (создание машины, клонирование репозитория, установка зависимостей, скачивание датасета, обучение, загрузка модели, остановка машины).

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-url` | str | **Да** | - | URL репозитория для клонирования |
| `--hf-dataset-repo` | str | **Да** | - | ID репозитория датасета на Hugging Face |
| `--hf-model-repo` | str | **Да** | - | ID репозитория модели на Hugging Face |
| `--machine-type` | str | Нет | `GPU+` | Тип машины |
| `--months` | int | Нет | `12` | Количество месяцев данных |
| `--hf-token` | str | Нет | `HF_TOKEN` | Hugging Face токен |
| `--api-key` | str | Нет | `PAPERSPACE_API_KEY` | Paperspace API ключ |

#### `ps-stop-machine`
Останавливает машину на Paperspace.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--machine-id` | str | **Да** | - | ID машины |
| `--api-key` | str | Нет | `PAPERSPACE_API_KEY` | Paperspace API ключ |

#### `ps-delete-machine`
Удаляет машину на Paperspace.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--machine-id` | str | **Да** | - | ID машины |
| `--api-key` | str | Нет | `PAPERSPACE_API_KEY` | Paperspace API ключ |

---

### Hugging Face: Загрузка данных

#### `hf-upload-ticks`
Загружает тиковые данные на Hugging Face Hub.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--ticks-dir` | str | Нет | `workspace/raw_data/ticks` | Директория с тиками |
| `--commit-message` | str | Нет | `Upload tick data` | Сообщение коммита |

#### `hf-upload-training`
Загружает данные для обучения на Hugging Face Hub.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--include-scalers` | flag | Нет | `True` | Включить scalers |
| `--no-scalers` | flag | Нет | - | Не включать scalers |
| `--include-cache` | flag | Нет | `False` | Включить кэши |
| `--commit-message` | str | Нет | `Upload training data` | Сообщение коммита |

#### `hf-upload-features`
Загружает результаты анализа фичей на Hugging Face Hub.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--analysis-dir` | str | Нет | `workspace/analysis-of-features` | Директория с результатами анализа |
| `--commit-message` | str | Нет | `Upload feature analysis results` | Сообщение коммита |

---

### Hugging Face: Скачивание данных

#### `hf-download-ticks`
Скачивает тиковые данные с Hugging Face Hub.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--local-dir` | str | Нет | `workspace/raw_data/ticks` | Локальная директория для сохранения |

#### `hf-download-training`
Скачивает данные для обучения с Hugging Face Hub.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--local-dir` | str | Нет | `workspace` | Локальная директория для сохранения |

#### `hf-download-features`
Скачивает результаты анализа фичей с Hugging Face Hub.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--local-dir` | str | Нет | `workspace/analysis-of-features` | Локальная директория для сохранения |

---

### Hugging Face: Удаление данных

#### `hf-delete-ticks`
Удаляет тиковые данные из Hugging Face датасета.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--commit-message` | str | Нет | `Delete tick data` | Сообщение коммита |

#### `hf-delete-training`
Удаляет данные для обучения из Hugging Face датасета.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--include-scalers` | flag | Нет | `True` | Удалять scalers |
| `--no-scalers` | flag | Нет | - | Не удалять scalers |
| `--include-cache` | flag | Нет | `True` | Удалять кэши |
| `--no-cache` | flag | Нет | - | Не удалять кэши |
| `--commit-message` | str | Нет | `Delete training data` | Сообщение коммита |

#### `hf-delete-features`
Удаляет результаты анализа фичей из Hugging Face датасета.

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--commit-message` | str | Нет | `Delete feature analysis results` | Сообщение коммита |

#### `hf-delete-all`
**⚠️ ОПАСНО:** Удаляет все данные из Hugging Face датасета (требует подтверждения).

| Опция | Тип | Обязательно | По умолчанию | Описание |
|-------|-----|-------------|--------------|----------|
| `--repo-id` | str | **Да** | - | ID репозитория (username/dataset-name) |
| `--token` | str | Нет | `HF_TOKEN` env var | Hugging Face токен |
| `--commit-message` | str | Нет | `Delete all dataset data` | Сообщение коммита |

**Примечание:** Команда требует ввода `yes` для подтверждения удаления.

---

## См. также

- [Настройка Paperspace](13_PAPERSPACE_SETUP.md) - подробное руководство по Paperspace
- [Оптимизация фичей](11_FEATURE_OPTIMIZATION.md) - анализ фичей
- [Подготовка данных](03_DATA_PREPARATION.md) - подготовка данных для обучения

