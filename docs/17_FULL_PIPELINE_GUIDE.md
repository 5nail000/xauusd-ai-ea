# Руководство по full_pipeline.py

## Обзор

`full_pipeline.py` - единый скрипт для автоматизации полного цикла работы с моделями: от подготовки данных до бэктестинга. Этот документ описывает структуру скрипта и предоставляет команды для ручного выполнения каждого этапа.

## Структура full_pipeline.py

Скрипт выполняет следующие этапы последовательно:

```
┌─────────────────────────────────────────────────────────────┐
│ ЭТАП 1: Подготовка данных                                  │
│   - Загрузка данных из MT5                                  │
│   - Генерация фичей                                         │
│   - Разделение на train/val/test                            │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ ОПЦИОНАЛЬНО: Оптимизация фичей                             │
│   - Удаление высококоррелированных фичей                   │
│   - Комплексный анализ фичей                                │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ ЭТАП 2: Обучение моделей                                    │
│   Вариант A: Обычное обучение                               │
│   Вариант B: Walk-Forward Validation                        │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│ ЭТАП 3: Бэктестинг                                          │
│   - Тестирование на test данных                             │
│   - Сохранение результатов                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Команды для ручного выполнения этапов

### ЭТАП 1: Подготовка данных

#### Базовая подготовка (6 месяцев, по умолчанию)

```bash
python prepare_gold_data.py
```

#### С указанием периода

```bash
# 12 месяцев
python prepare_gold_data.py --months 12

# 6 месяцев
python prepare_gold_data.py --months 6

# 30 дней
python prepare_gold_data.py --days 30

# 7 дней
python prepare_gold_data.py --days 7
```

#### С дополнительными параметрами

```bash
# Без тиковых данных (быстрее)
python prepare_gold_data.py --months 6 --no-ticks

# Без старших таймфреймов
python prepare_gold_data.py --months 6 --no-higher-tf

# Без тиков и старших таймфреймов (максимально быстро)
python prepare_gold_data.py --months 6 --no-ticks --no-higher-tf

# Принудительная регенерация (игнорировать кэш)
python prepare_gold_data.py --months 6 --force

# Без использования кэша
python prepare_gold_data.py --months 6 --no-cache

# Режим offline (только кэшированные данные, без MT5)
python prepare_gold_data.py --months 6 --offline

# Автоматически загружать сохраненные данные (не спрашивать)
python prepare_gold_data.py --months 6 --no-ask
```

#### Результаты этапа 1

После выполнения создаются файлы:
- `workspace/prepared/features/gold_train.csv`
- `workspace/prepared/features/gold_val.csv`
- `workspace/prepared/features/gold_test.csv`

---

### ОПЦИОНАЛЬНО: Оптимизация фичей

#### Удаление высококоррелированных фичей

**Важно:** Анализ выполняется на объединенном датасете (train+val+test) для гарантии одинакового набора фичей во всех файлах.

```bash
# С порогом корреляции 0.95 (по умолчанию)
python full_pipeline.py --skip-prepare --skip-train --skip-backtest --remove-correlated

# С кастомным порогом корреляции
python full_pipeline.py --skip-prepare --skip-train --skip-backtest --remove-correlated --correlation-threshold 0.90

# С порогом 0.85 (более агрессивное удаление)
python full_pipeline.py --skip-prepare --skip-train --skip-backtest --remove-correlated --correlation-threshold 0.85
```

**Результаты:**
- Список удаленных фичей: `workspace/prepared/features/features_to_remove_threshold_*.csv`
- Резервные копии: `*_backup.csv`
- Обновленные CSV файлы без коррелированных фичей

#### Комплексный анализ фичей

```bash
# Базовый анализ (без графиков)
python analyze_features_comprehensive.py \
    --train workspace/prepared/features/gold_train.csv \
    --val workspace/prepared/features/gold_val.csv \
    --test workspace/prepared/features/gold_test.csv \
    --target signal_class \
    --output-dir workspace/analysis-of-features \
    --top-features 50

# С генерацией графиков
python analyze_features_comprehensive.py \
    --train workspace/prepared/features/gold_train.csv \
    --val workspace/prepared/features/gold_val.csv \
    --test workspace/prepared/features/gold_test.csv \
    --target signal_class \
    --output-dir workspace/analysis-of-features \
    --top-features 50 \
    --generate-plots
```

**Результаты:**
- `workspace/analysis-of-features/feature_statistics.csv` - базовая статистика
- `workspace/analysis-of-features/feature_importance.csv` - важность фичей
- `workspace/analysis-of-features/outliers_analysis.csv` - анализ выбросов
- `workspace/analysis-of-features/feature_by_class_statistics.csv` - статистика по классам
- `workspace/analysis-of-features/feature_analysis_report.html` - HTML отчет
- `workspace/excluded_features.txt` - список фичей для исключения (создается автоматически)
- `workspace/analysis-of-features/plots/` - графики (если `--generate-plots`)

---

### ЭТАП 2: Обучение моделей

#### Вариант A: Обычное обучение

##### Обучить обе модели (encoder и timeseries)

```bash
# С параметрами по умолчанию (12 месяцев, 100 эпох, batch_size=32)
python train_all_models.py

# 6 месяцев данных
python train_all_models.py --months 6

# С кастомными параметрами
python train_all_models.py --months 6 --batch-size 32 --epochs 100 --patience 10
```

##### Только encoder модель

```bash
# Базовое обучение
python train_all_models.py --months 6 --encoder-only

# С параметрами
python train_all_models.py --months 6 --encoder-only --batch-size 32 --epochs 100 --patience 10
```

##### Только timeseries модель

```bash
python train_all_models.py --months 6 --timeseries-only
```

##### С дополнительными параметрами

```bash
# Без весов классов
python train_all_models.py --months 6 --encoder-only --no-class-weights

# С другим методом весов
python train_all_models.py --months 6 --encoder-only --class-weight-method inverse

# С W&B логированием
python train_all_models.py --months 6 --encoder-only --use-wandb
```

**Результаты этапа 2A:**
- `workspace/models/checkpoints/encoder_model.pth` - веса модели
- `workspace/models/checkpoints/timeseries_model.pth` - веса модели (если обучена)
- `workspace/prepared/scalers/feature_scaler_encoder.pkl` - scaler
- `workspace/prepared/scalers/feature_scaler_timeseries.pkl` - scaler (если обучена)
- `workspace/models/metrics/*_history.csv` - история обучения
- `workspace/models/metrics/*_training_curves.png` - графики обучения
- `workspace/analysis-of-train/*/` - детальные логи обучения

#### Быстрая проверка направления обучения

**Важно:** Хотя этот этап не включен в основной `full_pipeline.py`, вы можете использовать скрипт `utils/quick_test_training.py` для быстрой проверки направления обучения перед полным обучением на большом объеме данных.

**Когда использовать:**
- ✅ Нужно быстро проверить, в правильном ли направлении идет обучение
- ✅ Хотите оценить переобучение за 2-3 часа вместо 6+ часов
- ✅ Тестируете новые параметры или фичи
- ✅ Проверяете, стоит ли запускать полное обучение на 6+ месяцах

**Преимущества быстрой проверки:**
- ⚡ **В 40-50 раз быстрее** полного обучения
- ⚡ **Упрощенная модель** (d_model=128, n_layers=2 вместо 256, 4)
- ⚡ **Меньше данных** (1 месяц вместо 6)
- ⚡ **Короткие последовательности** (30 вместо 60)
- ⚡ **Больший batch_size** (64 вместо 32)
- ⚡ **Меньше эпох** (20 вместо 100)

**Команды:**

```bash
# Базовая быстрая проверка (1 месяц, 20 эпох)
python utils/quick_test_training.py

# 2 месяца данных для более надежной проверки
python utils/quick_test_training.py --months 2

# Меньший batch (если не хватает памяти)
python utils/quick_test_training.py --batch-size 32

# Меньше эпох для еще более быстрой проверки
python utils/quick_test_training.py --epochs 15

# Все параметры
python utils/quick_test_training.py \
    --months 1 \
    --batch-size 64 \
    --epochs 20 \
    --sequence-length 30 \
    --patience 5
```

**Что проверяет скрипт:**
1. **Сходимость:** Снижается ли loss?
2. **Переобучение:** Растет ли gap между train и val?
3. **Направление:** Улучшаются ли метрики?
4. **Стабильность:** Стабильны ли результаты?

**Результаты быстрой проверки:**
- `workspace/models/checkpoints/encoder_model_quick_test.pth` - веса модели
- `workspace/prepared/scalers/feature_scaler_quick_test.pkl` - scaler
- `workspace/models/metrics/encoder_model_quick_test_history.csv` - история обучения
- `workspace/models/metrics/encoder_model_quick_test_training_curves.png` - графики обучения
- `workspace/models/metrics/confusion_matrix_quick_test.png` - confusion matrix
- Автоматические рекомендации по результатам (переобучение, низкая точность и т.д.)

**Оценка времени:**
- Полное обучение (6 месяцев): ~35 мин/эпоха × 10 эпох = ~6 часов
- Быстрая проверка (1 месяц): ~7-10 мин/эпоха × 20 эпох = ~2-3 часа

**Рекомендации после быстрой проверки:**
- Если обнаружено переобучение (gap < -5%): увеличить dropout, добавить weight_decay
- Если низкая точность (< 30%): проверить качество данных, баланс классов
- Если хорошие результаты (> 50%): можно переходить к полному обучению

**Примечание:** Результаты быстрой проверки не заменяют полное обучение, но помогают быстро оценить направление и избежать долгого ожидания при проблемах.

#### Вариант B: Walk-Forward Validation

**Что такое Walk-Forward Validation?**

Walk-Forward Validation - это более реалистичный метод оценки модели для временных рядов. Вместо одного разделения на train/val/test, он создает несколько "окон" (fold'ов), которые движутся по времени:

1. **Первое окно**: обучается на данных [0..train], валидируется на [train..train+val], тестируется на [train+val..train+val+test]
2. **Второе окно**: сдвигается на `step_days` дней вперед
3. **И так далее...**

Это симулирует реальную торговлю, где модель обучается на прошлом и тестируется на будущем.

**Преимущества:**
- ✅ Более реалистичная оценка производительности
- ✅ Проверка стабильности модели во времени
- ✅ Выявление переобучения на конкретных периодах
- ✅ Агрегированные метрики по всем fold'ам

**Недостатки:**
- ⚠️ Медленнее (обучение выполняется N раз, где N - количество fold'ов)
- ⚠️ Требует больше данных
- ⚠️ Более сложная интерпретация результатов

**Команды:**

```bash
# С автоматическим вычислением параметров (рекомендуется)
python full_pipeline.py --skip-prepare --use-walk-forward --model-type encoder

# С указанными параметрами окон
python full_pipeline.py \
    --skip-prepare \
    --use-walk-forward \
    --model-type encoder \
    --walk-forward-train-days 60 \
    --walk-forward-val-days 15 \
    --walk-forward-test-days 15 \
    --walk-forward-step-days 10

# С кастомными параметрами обучения
python full_pipeline.py \
    --skip-prepare \
    --use-walk-forward \
    --model-type encoder \
    --epochs 50 \
    --batch-size 16 \
    --patience 5
```

**Автоматический расчет параметров:**

Если параметры не указаны, система автоматически вычисляет оптимальные значения на основе доступного периода данных:

- **Минимальные требования**: train=30 дней, val=7 дней, test=7 дней, step=5 дней
- **Максимальные значения**: train=120 дней, val=30 дней, test=30 дней, step=30 дней
- **Целевое количество fold'ов**: 3-10 (зависит от объема данных)
- **Пропорции**: train ~55%, val ~17%, test ~17%, step ~12% от окна

**Результаты этапа 2B:**
- `workspace/results/walk_forward/walk_forward_results.csv` - детальные результаты по каждому fold'у:
  - Метрики для каждого fold'а (accuracy, precision, recall, F1)
  - Даты начала и конца каждого окна
  - Агрегированные статистики (mean, std, min, max)
- `workspace/models/checkpoints/{model_type}_model.pth` - последняя обученная модель (для бэктестинга)
- Консольный вывод с summary всех fold'ов

**Анализ результатов:**

После выполнения Walk-Forward Validation рекомендуется:

1. Открыть `walk_forward_results.csv` и проанализировать:
   - Стабильность метрик между fold'ами
   - Тренды (улучшается/ухудшается ли модель со временем)
   - Выбросы (fold'ы с аномально плохими/хорошими результатами)

2. Выбрать лучшую модель:
   - По средним метрикам
   - По стабильности (низкое std)
   - По последним fold'ам (более релевантные для будущего)

3. Использовать выбранную модель для бэктестинга или продакшена

---

### ЭТАП 3: Бэктестинг

#### Базовый бэктестинг

```bash
# С encoder моделью (по умолчанию)
python backtest_strategy.py

# С timeseries моделью
python backtest_strategy.py --model-type timeseries
```

#### С кастомными параметрами

```bash
# С указанием модели и scaler
python backtest_strategy.py \
    --model-path workspace/models/checkpoints/encoder_model.pth \
    --scaler-path workspace/prepared/scalers/feature_scaler_encoder.pkl \
    --model-type encoder
```

**Результаты этапа 3:**
- `trading/backtest_results.csv` - основные метрики
- `trading/equity_history.csv` - история equity
- `trading/closed_positions.csv` - все закрытые позиции

---

## Полный цикл через full_pipeline.py

### Базовый запуск

```bash
# Полный цикл с параметрами по умолчанию (12 месяцев)
python full_pipeline.py
```

### С параметрами

```bash
# 6 месяцев, только encoder модель
python full_pipeline.py --months 6 --encoder-only

# 12 месяцев, обе модели, с удалением коррелированных фичей
python full_pipeline.py --months 12 --remove-correlated

# С анализом фичей
python full_pipeline.py --months 12 --analyze-features --generate-feature-plots

# С кастомными параметрами обучения
python full_pipeline.py --months 6 --encoder-only --epochs 50 --batch-size 16
```

### Пропуск этапов

```bash
# Пропустить подготовку данных (если уже есть)
python full_pipeline.py --skip-prepare

# Пропустить обучение (если модель уже обучена)
python full_pipeline.py --skip-train

# Только бэктестинг
python full_pipeline.py --skip-prepare --skip-train

# Только обучение и бэктестинг
python full_pipeline.py --skip-prepare
```

### Walk-Forward Validation вместо обычного обучения

**Когда использовать Walk-Forward Validation:**

- ✅ Финальная валидация перед продакшеном
- ✅ Нужна более реалистичная оценка производительности
- ✅ Достаточно данных (минимум 60-90 дней для 3-5 fold'ов)
- ✅ Достаточно времени (обучение выполняется N раз)

**Когда использовать обычное обучение:**

- ✅ Быстрое тестирование и эксперименты
- ✅ Ограниченные ресурсы (время/GPU)
- ✅ Первичная оценка модели
- ✅ Мало данных (< 60 дней)

**Команды:**

```bash
# С автоматическими параметрами (рекомендуется)
python full_pipeline.py --months 6 --use-walk-forward --encoder-only

# С указанными параметрами
python full_pipeline.py \
    --months 6 \
    --use-walk-forward \
    --encoder-only \
    --walk-forward-train-days 60 \
    --walk-forward-val-days 15 \
    --walk-forward-test-days 15 \
    --walk-forward-step-days 10

# Полный цикл с Walk-Forward Validation
python full_pipeline.py \
    --months 12 \
    --use-walk-forward \
    --encoder-only \
    --epochs 50 \
    --batch-size 16
```

**Оценка времени выполнения:**

Если обычное обучение занимает T времени, то Walk-Forward Validation займет примерно:
- **Время = T × N**, где N - количество fold'ов
- Например: если обучение занимает 1 час и создается 5 fold'ов, то Walk-Forward займет ~5 часов
- Количество fold'ов зависит от объема данных и параметров окон

---

## Типичные сценарии использования

### Сценарий 1: Быстрая проверка направления перед полным обучением

```bash
# 1. Подготовка данных (1 месяц для быстрой проверки)
python prepare_gold_data.py --months 1

# 2. Быстрая проверка направления (2-3 часа вместо 6+ часов)
python utils/quick_test_training.py

# 3. Анализ результатов быстрой проверки
# Проверить workspace/models/metrics/encoder_model_quick_test_history.csv
# Проверить workspace/models/metrics/encoder_model_quick_test_training_curves.png
# Оценить рекомендации в консольном выводе

# 4. Если результаты хорошие - запустить полное обучение
python train_all_models.py --months 6 --encoder-only --epochs 100

# 5. Бэктестинг
python backtest_strategy.py
```

### Сценарий 1A: Быстрый тест на малом объеме данных

```bash
# 1. Подготовка данных (7 дней)
python prepare_gold_data.py --days 7 --no-ticks --no-higher-tf

# 2. Обучение encoder модели
python train_all_models.py --months 1 --encoder-only --epochs 20

# 3. Бэктестинг
python backtest_strategy.py
```

### Сценарий 2: Полная подготовка с оптимизацией

```bash
# 1. Подготовка данных
python prepare_gold_data.py --months 6

# 2. Удаление коррелированных фичей
python full_pipeline.py --skip-prepare --skip-train --skip-backtest --remove-correlated --correlation-threshold 0.90

# 3. Анализ фичей
python analyze_features_comprehensive.py \
    --train workspace/prepared/features/gold_train.csv \
    --val workspace/prepared/features/gold_val.csv \
    --test workspace/prepared/features/gold_test.csv \
    --target signal_class \
    --output-dir workspace/analysis-of-features \
    --top-features 50 \
    --generate-plots

# 4. Обучение моделей
python train_all_models.py --months 6 --encoder-only --epochs 100

# 5. Бэктестинг
python backtest_strategy.py
```

### Сценарий 3: Walk-Forward Validation для финальной валидации

```bash
# 1. Подготовка данных (нужно больше данных для Walk-Forward)
python prepare_gold_data.py --months 12

# 2. Walk-Forward Validation с автоматическими параметрами
python full_pipeline.py \
    --skip-prepare \
    --use-walk-forward \
    --encoder-only \
    --epochs 100 \
    --batch-size 32

# 3. Анализ результатов Walk-Forward
# Открыть workspace/results/walk_forward/walk_forward_results.csv
# Проанализировать:
#   - Стабильность метрик между fold'ами (низкое std = хорошо)
#   - Тренды (улучшается/ухудшается ли модель со временем)
#   - Выбросы (fold'ы с аномально плохими/хорошими результатами)

# 4. Бэктестинг на последней модели (или выбрать лучшую по метрикам)
python backtest_strategy.py

# Альтернатива: выбрать лучшую модель из Walk-Forward результатов
# и использовать её для бэктестинга
```

**Что анализировать в результатах Walk-Forward:**

1. **Стабильность метрик:**
   - Низкое стандартное отклонение (std) = модель стабильна
   - Высокое std = модель нестабильна, возможно переобучение

2. **Тренды:**
   - Улучшение метрик со временем = модель адаптируется
   - Ухудшение метрик = возможен дрифт данных

3. **Выбросы:**
   - Fold'ы с аномально плохими результатами могут указывать на проблемные периоды
   - Fold'ы с аномально хорошими результатами могут указывать на переобучение

### Сценарий 4: Пересчет фичей уровней (support/resistance)

```bash
# 1. Удалить старые колонки уровней из CSV
python utils/temp_remove_level_features.py

# 2. Пересчитать только фичи уровней (остальные останутся без изменений)
python prepare_gold_data.py --months 6 --no-ask
```

### Сценарий 5: Обучение на Paperspace

```bash
# 1. Локально: подготовка данных
python prepare_gold_data.py --months 12

# 2. Локально: создание архива для Paperspace
python cloud_services.py upload-training --host paperspace.com --path /storage/

# 3. На Paperspace: загрузка данных
# (выполнить на Paperspace)
python cloud_services.py download-training --host paperspace.com --path /storage/

# 4. На Paperspace: обучение
python train_all_models.py --months 12 --encoder-only

# 5. На Paperspace: скачивание результатов
# (выполнить на Paperspace)
python cloud_services.py download-training --host paperspace.com --path /storage/ --download-results
```

### Сценарий 6: Использование Hugging Face

```bash
# 1. Локально: подготовка данных
python prepare_gold_data.py --months 6

# 2. Локально: загрузка данных на Hugging Face
python cloud_services.py hf-upload-training --repo-id username/dataset-name

# 3. На удаленной машине: скачивание данных
python cloud_services.py hf-download-training --repo-id username/dataset-name

# 4. На удаленной машине: обучение
python train_all_models.py --months 6 --encoder-only

# 5. Локально: скачивание результатов (если загружены обратно)
python cloud_services.py hf-download-training --repo-id username/dataset-name
```

---

## Параметры командной строки

### prepare_gold_data.py

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `-m, --months` | Количество месяцев данных | 6 |
| `-d, --days` | Количество дней данных (приоритет над --months) | None |
| `--symbol` | Торговый символ | XAUUSD |
| `--no-ticks` | Не загружать тиковые данные | False |
| `--no-higher-tf` | Не загружать старшие таймфреймы | False |
| `--force` | Принудительно регенерировать данные | False |
| `--no-cache` | Не использовать кэш | False |
| `--offline` | Режим offline (без MT5) | False |
| `--no-ask` | Не спрашивать при наличии данных | False |

### train_all_models.py

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `-m, --months` | Количество месяцев данных (для метаданных) | 12 |
| `--encoder-only` | Обучить только encoder модель | False |
| `--timeseries-only` | Обучить только timeseries модель | False |
| `--batch-size` | Размер батча | 32 |
| `--epochs` | Количество эпох | 100 |
| `--patience` | Терпение для early stopping | 10 |
| `--dropout` | Dropout rate для регуляризации | 0.1 |
| `--learning-rate` | Learning rate | 1e-4 |
| `--weight-decay` | Weight decay для регуляризации | 1e-5 |
| `--no-class-weights` | Не использовать веса классов | False |
| `--class-weight-method` | Метод весов (balanced/inverse/sqrt) | balanced |
| `--use-wandb` | Использовать Weights & Biases | False |
| `--wandb-project` | Название проекта W&B | xauusd-ai-ea |

### full_pipeline.py

Все параметры из `prepare_gold_data.py` и `train_all_models.py`, плюс:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--remove-correlated` | Удалить высококоррелированные фичи | False |
| `--correlation-threshold` | Порог корреляции | 0.95 |
| `--analyze-features` | Комплексный анализ фичей | False |
| `--generate-feature-plots` | Генерировать графики при анализе | False |
| `--model-type` | Тип модели для бэктестинга (encoder/timeseries) | encoder |
| `--skip-prepare` | Пропустить подготовку данных | False |
| `--skip-train` | Пропустить обучение | False |
| `--skip-backtest` | Пропустить бэктестинг | False |
| `--use-walk-forward` | Использовать Walk-Forward Validation вместо обычного обучения | False |
| `--walk-forward-train-days` | Размер обучающего окна в днях (auto = вычисляется автоматически) | auto |
| `--walk-forward-val-days` | Размер валидационного окна в днях (auto = вычисляется автоматически) | auto |
| `--walk-forward-test-days` | Размер тестового окна в днях (auto = вычисляется автоматически) | auto |
| `--walk-forward-step-days` | Шаг сдвига окна в днях (auto = вычисляется автоматически) | auto |

**Примечание:** Если параметры не указаны, система автоматически вычисляет оптимальные значения на основе доступного периода данных. Рекомендуется использовать автоматический расчет для первого запуска.

---

## Проверка результатов

### Проверка подготовленных данных

```bash
# Проверить наличие файлов
ls workspace/prepared/features/gold_*.csv

# Проверить размеры файлов
python -c "import pandas as pd; print('Train:', len(pd.read_csv('workspace/prepared/features/gold_train.csv', index_col=0))); print('Val:', len(pd.read_csv('workspace/prepared/features/gold_val.csv', index_col=0))); print('Test:', len(pd.read_csv('workspace/prepared/features/gold_test.csv', index_col=0)))"
```

### Проверка обученных моделей

```bash
# Проверить наличие моделей
ls workspace/models/checkpoints/*.pth
ls workspace/prepared/scalers/*.pkl

# Проверить метрики обучения
ls workspace/models/metrics/*.csv
ls workspace/models/metrics/*.png
```

### Проверка результатов бэктестинга

```bash
# Проверить результаты
ls trading/*.csv

# Просмотреть основные метрики
python -c "import pandas as pd; print(pd.read_csv('trading/backtest_results.csv'))"
```

---

## Устранение проблем

### Проблема: Данные уже существуют

**Решение:** Используйте `--no-ask` для автоматической загрузки или `--force` для пересоздания.

```bash
python prepare_gold_data.py --months 6 --no-ask
# или
python prepare_gold_data.py --months 6 --force
```

### Проблема: Модель уже обучена

**Решение:** Используйте `--skip-train` или удалите старую модель.

```bash
python full_pipeline.py --skip-prepare --skip-train
```

### Проблема: Недостаточно памяти

**Решение:** Уменьшите batch_size или используйте меньше данных.

```bash
python train_all_models.py --months 6 --encoder-only --batch-size 16
```

### Проблема: Ошибка при загрузке данных из MT5

**Решение:** Используйте режим offline с кэшированными данными.

```bash
python prepare_gold_data.py --months 6 --offline
```

---

## Дополнительные ресурсы

- [03_DATA_PREPARATION.md](03_DATA_PREPARATION.md) - Детальное описание подготовки данных
- [04_TRAINING_GUIDE.md](04_TRAINING_GUIDE.md) - Детальное описание обучения
- [06_BACKTESTING_GUIDE.md](06_BACKTESTING_GUIDE.md) - Детальное описание бэктестинга
- [11_FEATURE_OPTIMIZATION.md](11_FEATURE_OPTIMIZATION.md) - Оптимизация фичей
- [15_CLOUD_SERVICES.md](15_CLOUD_SERVICES.md) - Работа с облачными сервисами
- [16_TRAINING_PREPARATION_GUIDE.md](16_TRAINING_PREPARATION_GUIDE.md) - Подготовка к обучению

---

**Последнее обновление:** Структура соответствует `full_pipeline.py` версии с поддержкой `excluded_features.txt` и `analysis-of-features`

