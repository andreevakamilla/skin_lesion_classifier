# Классификатор кожных заболеваний

**Андреева Камилла**

## Постановка задачи

Разработать систему компьютерного зрения для классификации кожных заболеваний по дерматоскопическим изображениям на 7 типов диагнозов. Ранняя детекция злокачественных образований критически важна — при раннем обнаружении меланомы 5-летняя выживаемость превышает 95%, при позднем падает до 20%. Система служит инструментом скрининга, помогая выделить подозрительные случаи для приоритетного осмотра дерматологом.

## Формат входных и выходных данных

**Вход:** тензор изображения формата RGB размерности `[1, 3, 224, 224]` (batch, channels, height, width), нормализованный по статистикам ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

Пример входного изображения из датасета: [ISIC_0024306.jpg](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Выход:** JSON со следующими полями:

```json
{
  "predicted_class": "mel",
  "predicted_class_full": "Меланома",
  "confidence": 0.82,
  "all_probabilities": {
    "akiec": 0.03,
    "bcc": 0.02,
    "bkl": 0.05,
    "df": 0.01,
    "mel": 0.82,
    "nv": 0.06,
    "vasc": 0.01
  },
  "top_3": [
    ["mel", 0.82],
    ["nv", 0.06],
    ["bkl", 0.05]
  ],
  "risk_flag": true
}
```

`risk_flag = true` если предсказан злокачественный класс (mel, bcc, akiec) или confidence < 0.6.

## Классы

| Код   | Название                    | Тип               | % в датасете |
| ----- | --------------------------- | ----------------- | ------------ |
| nv    | Меланоцитарный невус        | Доброкачественный | 67%          |
| mel   | Меланома                    | Злокачественный   | 11%          |
| bkl   | Доброкачественный кератоз   | Доброкачественный | 11%          |
| bcc   | Базальноклеточная карцинома | Злокачественный   | 5%           |
| akiec | Актинический кератоз        | Злокачественный   | 3%           |
| vasc  | Сосудистое поражение        | Доброкачественный | 1.4%         |
| df    | Дерматофиброма              | Доброкачественный | 1.1%         |

## Метрики

**Основные:** Balanced Accuracy и Macro F1-score — используются из-за сильного дисбаланса классов (67% — один класс). Обычная accuracy была бы misleading: модель предсказывающая всегда "невус" дала бы 67% accuracy.

**Критическая метрика:** Recall для класса melanoma ≥ 0.70 — пропуск меланомы опаснее ложного срабатывания.

**Ожидаемые значения**:

- Бейзлайн (SimpleCNN): Balanced Accuracy ~0.50, Macro F1 ~0.38
- Основная модель (EfficientNet-B1): Balanced Accuracy ~0.67, Macro F1 ~0.57

**Достигнутые результаты:**

| Метрика           | SimpleCNN (бейзлайн) | EfficientNet-B1 |
| ----------------- | -------------------- | --------------- |
| Accuracy          | 0.55                 | **0.72**        |
| Balanced Accuracy | 0.48                 | **0.65**        |
| Macro F1          | 0.38                 | **0.54**        |
| mel recall        | 0.60                 | **0.59**        |
| nv recall         | 0.58                 | **0.76**        |

## Валидация и тест

Использован **Stratified Group Split** по `lesion_id` — все снимки одного поражения попадают в один сплит, что исключает утечку данных (в HAM10000 есть несколько фото одного поражения). Random seed зафиксирован (42) для воспроизводимости.

- Train: 70% (6959 сэмплов)
- Validation: 15% (1529 сэмплов)
- Test: 15% (1527 сэмплов)

## Датасеты

**HAM10000** (Human Against Machine with 10000 training images) — опубликован в 2018 году Philipp Tschandl et al. ([Nature Medicine, 2019](https://www.nature.com/articles/s41591-018-0307-0)).

- Источник: [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/data), [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- Объём: ~1.3 ГБ
- Кол-во сэмплов: 10 015
- Кол-во классов: 7
- Исходный размер изображений: 600×450 px, RGB
- Размер на входе модели: 224×224 px (после ресайза)

**Особенности и сложности:**

- Сильный дисбаланс классов (невус 67%, дерматофиброма 1%)
- Артефакты на снимках: волосяной покров, чернильные метки, линейки, пузырьки воздуха
- Разное освещение и оборудование съёмки

## Моделирование

### Предобработка данных

Все изображения ресайзятся до 224×224 и нормализуются по статистикам ImageNet. Для обучения применяются аугментации:

- HorizontalFlip, VerticalFlip, RandomRotate90
- ShiftScaleRotate
- RandomBrightnessContrast / HueSaturationValue
- CoarseDropout (имитация артефактов)
- GaussNoise

Для валидации и теста — только ресайз и нормализация.

### Бейзлайн

**SimpleCNN** — 3 свёрточных блока (Conv2d + BatchNorm + ReLU + MaxPool) и 2 полносвязных слоя, обучается с нуля.

Пайплайн: ресайз 224×224 → нормализация → минимальные аугментации (только flip) -> обучение 35 эпох, AdamW, Focal Loss.

### Основная модель

**EfficientNet-B1** с предобученными весами ImageNet, поэтапная разморозка:

- Этап 1 (5 эпох): обучение только головы (backbone заморожен), lr=1e-3
- Этап 2 (20 эпох): разморозка последних 2 блоков, lr=5e-4
- Этап 3 (10 эпох): полная разморозка, lr=1e-5

Loss: **Focal Loss** (γ=3) — снижает вклад лёгких примеров, фокусирует обучение на редких классах. Оптимизатор: AdamW с weight decay 1e-4, cosine annealing.

**Постобработка:** logits -> softmax -> вероятности по 7 классам -> argmax -> risk_flag.

## Структура проекта

```
skin-lesion-classifier/
├── configs/
│   ├── config.yaml              # Главный конфиг Hydra
│   ├── data/ham10000.yaml       # Параметры данных и аугментаций
│   ├── model/
│   │   ├── efficientnet.yaml
│   │   └── simple_cnn.yaml
│   ├── training/default.yaml   # Гиперпараметры обучения
│   └── logging/mlflow.yaml     # MLflow конфиг
├── data/                        # Данные (управляются DVC)
│   ├── HAM10000_metadata.csv.dvc
│   ├── HAM10000_images_part_1.dvc
│   └── HAM10000_images_part_2.dvc
├── plots/                       # Графики обучения
├── skin_lesion_classifier/
│   ├── commands.py              # Единая точка входа CLI
│   ├── constants.py             # Глобальные константы
│   ├── data/
│   │   └── dataset.py           # Dataset, аугментации, DataModule
│   ├── training/
│   │   ├── lightning_module.py  # Lightning Module (модели, loss, метрики)
│   │   └── train.py             # Пайплайн обучения (Hydra entry point)
│   └── inference/
│       ├── predictor.py         # ONNX-based инференс
│       └── tensorrt_converter.py
├── test_serving.py              # Тест MLflow Serving
├── .pre-commit-config.yaml
├── .flake8
├── pyproject.toml
└── uv.lock
```

---

## Setup

### Установка

```bash
git clone https://github.com/andreevakamilla/skin_lesion_classifier.git
cd skin_lesion_classifier

pip install uv
uv sync
source .venv/bin/activate

pre-commit install
```

## Data

```bash
dvc pull

# Или через kaggle
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/

# Или встроенная функция (скачивает метаданные)
python skin_lesion_classifier/commands.py download-data
```

## Train

```bash
# Запуск MLflow сервера (в отдельном терминале)
mlflow server --host 127.0.0.1 --port 8080

# Основная модель (EfficientNet-B1)
python skin_lesion_classifier/training/train.py

# Бейзлайн (SimpleCNN)
python skin_lesion_classifier/training/train.py model=simple_cnn

# Переопределение параметров через Hydra
python skin_lesion_classifier/training/train.py training.batch_size=64 training.lr_head=5e-4

# Изменить MLflow сервер
python skin_lesion_classifier/training/train.py logging.mlflow_tracking_uri=http://my-server:5000
```

Конфиги находятся в `configs/`. Результаты экспериментов в MLflow UI: `http://127.0.0.1:8080`.

Графики обучения (loss, balanced accuracy, mel recall) сохранены в `plots/`.

## Production preparation

ONNX экспорт выполняется автоматически в конце обучения (`checkpoints/model.onnx`).

**Конвертация в TensorRT** (требует TensorRT >= 8.0):

```bash
python skin_lesion_classifier/commands.py convert-trt \
  --onnx-model checkpoints/model.onnx \
  --engine-path checkpoints/model.trt \
  --fp16
```

## Infer

Инференс использует ONNX Runtime:

```bash
python skin_lesion_classifier/commands.py infer \
  --image path/to/image.jpg \
  --onnx-model checkpoints/model.onnx
```

**Формат входных данных:** JPEG или PNG изображение кожного поражения, желательно дерматоскопическое. Минимальный размер 224×224 px.

**Пример вывода:**

```
Предсказание: Меланоцитарный невус (nv)
Уверенность:  0.8234
Risk flag:    НЕТ

Top-3:
  nv       0.8234
  mel      0.1102
  bkl      0.0412
```

### MLflow Serving

```bash
# Поднять inference server
mlflow models serve \
  -m "mlruns/<experiment_id>/models/<model_id>/artifacts" \
  --port 5002 \
  --no-conda

# Тестовый запрос
python test_serving.py
```

API endpoint: `POST http://127.0.0.1:5002/invocations`

```json
{
  "inputs": [[[...]]]
}
```

Ответ:

```json
{
  "predictions": {
    "logits": [[-0.42, -0.61, -0.59, 0.16, -1.24, -0.67, -0.65]]
  }
}
```
