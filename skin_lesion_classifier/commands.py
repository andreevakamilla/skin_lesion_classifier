import fire


def train() -> None:
    from skin_lesion_classifier.training.train import train as _train

    _train()


def infer(
    image: str,
    onnx_model: str = "checkpoints/model.onnx",
    image_size: int = 224,
) -> None:
    from skin_lesion_classifier.inference.predictor import SkinLesionPredictor

    predictor = SkinLesionPredictor(onnx_path=onnx_model, image_size=image_size)
    result = predictor.predict(image)

    print(f"\nПредсказание: {result['predicted_class_full']} ({result['predicted_class']})")
    print(f"Уверенность:  {result['confidence']:.4f}")
    print(f"Risk flag:    {'ДА' if result['risk_flag'] else 'НЕТ'}")
    print("\nTop-3:")
    for class_name, prob in result["top_3"]:
        print(f"  {class_name:<8} {prob:.4f}")


def download_data(data_dir: str = "data") -> None:
    from skin_lesion_classifier.data.dataset import download_data as _download

    _download(data_dir)


def main() -> None:
    fire.Fire(
        {
            "train": train,
            "infer": infer,
            "download-data": download_data,
        }
    )


if __name__ == "__main__":
    main()
