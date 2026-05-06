# skin-lesion-classifier

Классификатор кожных заболеваний по дерматоскопическим изображениям (HAM10000).


Классификация дерматоскопических изображений по 7 типам диагнозов. Ранняя детекция меланомы критически важна — при раннем обнаружении 5-летняя выживаемость превышает 95%.

| Код | Название |
|-----|----------|
| akiec | Актинический кератоз |
| bcc | Базальноклеточная карцинома |
| bkl | Доброкачественный кератоз |
| df | Дерматофиброма |
| mel | Меланома |
| nv | Меланоцитарный невус |
| vasc | Сосудистое поражение |


| Метрика | SimpleCNN (бейзлайн) | EfficientNet-B1 |
|---------|---------------------|-----------------|
| Accuracy | 0.55 | **0.72** |
| Balanced Accuracy | 0.48 | **0.67** |
| Macro F1 | 0.38 | **0.57** |
| mel recall | 0.60 | **0.72** |


```bash
pip install uv
uv sync

source .venv/bin/activate

pre-commit install
```


```bash
dvc pull

# Или через kaggle CLI
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d data/
```

```bash
mlflow server --host 127.0.0.1 --port 8082
```
В другом терминалн
```bash
# Основная модель (EfficientNet-B1)
python skin_lesion_classifier/training/train.py

# Бейзлайн (SimpleCNN)
python skin_lesion_classifier/training/train.py model=simple_cnn

# Кастомные параметры через Hydra
python skin_lesion_classifier/training/train.py training.batch_size=64
```

Конфиги в `configs/`. Логи экспериментов в MLflow — запусти перед обучением:



ONNX экспорт выполняется автоматически в конце обучения (`checkpoints/model.onnx`).

Для инференса нужны только:
- `checkpoints/model.onnx`
- `skin_lesion_classifier/inference/predictor.py`
- `skin_lesion_classifier/constants.py`

