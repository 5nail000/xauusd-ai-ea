# Руководство по облачным сервисам

## Обзор

Модуль `cloud_services.py` предоставляет утилиты для работы с облачными сервисами:
- **Paperspace** - для обучения моделей на мощных серверах
- **Hugging Face Hub** - для хранения и обмена данными, моделями и результатами анализа

## Установка зависимостей

```bash
# Для работы с Hugging Face
pip install huggingface_hub>=0.20.0

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

1. Настройте SSH доступ к вашему Paperspace серверу
2. Убедитесь, что у вас есть доступ к `/storage/` или другой директории

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
python full_pipeline.py --months 6 --analyze-features --generate-feature-plots

# Или отдельно
python analyze_features_comprehensive.py \
  --train workspace/prepared/features/gold_train.csv \
  --val workspace/prepared/features/gold_val.csv \
  --test workspace/prepared/features/gold_test.csv \
  --generate-plots
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
python full_pipeline.py --analyze-features --generate-feature-plots

# 2. Загрузить результаты на Hugging Face
python cloud_services.py hf-upload-features \
  --repo-id username/xauusd-feature-analysis

# 3. Другие пользователи могут скачать и просмотреть
python cloud_services.py hf-download-features \
  --repo-id username/xauusd-feature-analysis

# 4. Открыть HTML отчет
# workspace/analysis-of-features/feature_analysis_report.html
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
from cloud_services import HuggingFaceUploader, HuggingFaceDownloader

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
```

**Примечание:** Для обратной совместимости старые названия методов (без префиксов `paperspace_` и `hf_`) также доступны как алиасы, но рекомендуется использовать новые названия для ясности.

---

## См. также

- [Настройка Paperspace](13_PAPERSPACE_SETUP.md) - подробное руководство по Paperspace
- [Оптимизация фичей](11_FEATURE_OPTIMIZATION.md) - анализ фичей
- [Подготовка данных](03_DATA_PREPARATION.md) - подготовка данных для обучения

