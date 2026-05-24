import glob
import subprocess

import hydra
import mlflow
import onnx
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from skin_lesion_classifier.constants import CLASS_TO_IDX
from skin_lesion_classifier.data.dataset import HAM10000DataModule, compute_class_weights
from skin_lesion_classifier.training.lightning_module import SkinLesionModule


def get_git_commit_id() -> str:
    """Возвращает текущий git commit id."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> None:
    """Полный пайплайн обучения: данные, модель, логирование, тест, ONNX экспорт."""
    subprocess.run(["dvc", "pull"], check=False)
    pl.seed_everything(cfg.data.random_seed, workers=True)

    data_module = HAM10000DataModule(cfg.data)
    data_module.setup()

    class_weights = compute_class_weights(
        data_module.df_train["label"].values,
        num_classes=len(CLASS_TO_IDX),
    )

    model = SkinLesionModule(cfg, class_weights=class_weights)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.mlflow_tracking_uri,
        log_model=True,
    )
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", get_git_commit_id())

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="best-{epoch:02d}-{val_balanced_accuracy:.4f}",
        monitor="val/balanced_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
        save_on_train_epoch_end=False,
    )
    early_stopping = EarlyStopping(
        monitor="val/balanced_accuracy",
        patience=cfg.training.patience,
        min_delta=cfg.training.min_delta,
        mode="max",
    )

    total_epochs = (
        cfg.training.epochs_head + cfg.training.epochs_finetune + cfg.training.epochs_full
    )
    trainer = pl.Trainer(
        max_epochs=total_epochs,
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping],
        log_every_n_steps=cfg.logging.log_every_n_steps,
        deterministic=True,
        devices=1,
        accelerator="gpu",
        num_sanity_val_steps=0,
    )

    trainer.fit(model, datamodule=data_module)

    manual_ckpts = glob.glob(f"{cfg.training.checkpoint_dir}/manual_best_*.ckpt")
    best_ckpt = max(
        manual_ckpts,
        key=lambda x: float(x.split("bal_acc=")[1].replace(".ckpt", "")),
    )
    print(f"Загружаем лучший чекпоинт: {best_ckpt}")
    trainer.test(model, datamodule=data_module, ckpt_path=best_ckpt)

    best_model = SkinLesionModule.load_from_checkpoint(best_ckpt, cfg=cfg)
    best_model.export_onnx(cfg.training.onnx_export_path, image_size=cfg.data.image_size)

    mlflow.set_tracking_uri(cfg.logging.mlflow_tracking_uri)
    with mlflow.start_run(run_id=mlflow_logger.run_id, nested=True):
        mlflow.onnx.log_model(
            onnx_model=onnx.load(cfg.training.onnx_export_path),
            artifact_path="onnx_model",
        )
        print("[MLFLOW] ONNX модель залогирована")


if __name__ == "__main__":
    train()
