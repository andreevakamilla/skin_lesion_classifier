import numpy as np
import requests
import torch

from skin_lesion_classifier.constants import CLASS_NAMES


def test_mlflow_serving(host: str = "127.0.0.1", port: int = 5002) -> None:
    url = f"http://{host}:{port}/invocations"
    data = np.random.randn(1, 3, 224, 224).astype(np.float32).tolist()
    response = requests.post(
        url,
        json={"inputs": data},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200, f"Ошибка: {response.status_code}"
    logits = response.json()["predictions"]["logits"][0]
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()
    predicted_class = CLASS_NAMES[int(np.argmax(probs))]
    print(f"Status: {response.status_code}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {float(np.max(probs)):.4f}")


if __name__ == "__main__":
    test_mlflow_serving()
