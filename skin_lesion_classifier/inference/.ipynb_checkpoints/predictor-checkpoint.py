from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from skin_lesion_classifier.constants import (
    CLASS_FULL_NAMES,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MALIGNANT_CLASSES,
)


def preprocess_image(image_path: Union[str, Path], image_size: int = 224) -> np.ndarray:
    """Предобрабатывает изображение для инференса."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size), Image.BILINEAR)
    image_array = np.array(image, dtype=np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image_array = (image_array - mean) / std

    return image_array.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Вычисляет softmax."""
    exp_logits = np.exp(logits - logits.max())
    return exp_logits / exp_logits.sum()


class SkinLesionPredictor:
    """Предсказатель на основе ONNX Runtime."""

    def __init__(self, onnx_path: Union[str, Path], image_size: int = 224) -> None:
        self.image_size = image_size
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: Union[str, Path]) -> dict:
        input_array = preprocess_image(image_path, self.image_size)
        logits = self.session.run(None, {self.input_name: input_array})[0][0]
        probabilities = softmax(logits)

        predicted_idx = int(probabilities.argmax())
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        top3_indices = probabilities.argsort()[::-1][:3]
        top_3 = [(CLASS_NAMES[idx], float(probabilities[idx])) for idx in top3_indices]

        return {
            "predicted_class": predicted_class,
            "predicted_class_full": CLASS_FULL_NAMES[predicted_class],
            "confidence": confidence,
            "all_probabilities": {
                CLASS_NAMES[idx]: float(probabilities[idx]) for idx in range(len(CLASS_NAMES))
            },
            "top_3": top_3,
            "risk_flag": predicted_class in MALIGNANT_CLASSES or confidence < 0.6,
        }
