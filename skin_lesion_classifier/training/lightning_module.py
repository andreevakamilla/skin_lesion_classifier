from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig

from skin_lesion_classifier.constants import CLASS_NAMES


class SimpleCNN(nn.Module):
    """Бейзлайн: 3 conv блока + 2 FC слоя."""

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class SkinLesionEfficientNet(nn.Module):
    """EfficientNet-B1 с кастомной головой и поэтапной разморозкой."""

    def __init__(self, num_classes: int, pretrained: bool, dropout: float) -> None:
        super().__init__()
        weights = models.EfficientNet_B1_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.efficientnet_b1(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def freeze_backbone(self) -> None:
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_from(self, block_idx: int) -> None:
        blocks = list(self.backbone.features.children())
        start = len(blocks) + block_idx if block_idx < 0 else block_idx
        for idx, block in enumerate(blocks):
            if idx >= start:
                for param in block.parameters():
                    param.requires_grad = True

    def unfreeze_all(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, lr_backbone: float, lr_head: float) -> list:
        backbone_params = [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and "classifier" not in name
        ]
        head_params = [
            param
            for name, param in self.named_parameters()
            if param.requires_grad and "classifier" in name
        ]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_head})
        return groups

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


class FocalLoss(nn.Module):
    """Focal Loss для работы с дисбалансом классов."""

    def __init__(self, gamma: float, alpha: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        prob_t = torch.exp(-ce_loss)
        focal_weight = (1 - prob_t) ** self.gamma

        if self.alpha is not None:
            focal_weight = self.alpha.to(inputs.device)[targets] * focal_weight

        return (focal_weight * ce_loss).mean()


class SkinLesionModule(pl.LightningModule):
    """Lightning Module с поэтапной разморозкой."""

    def __init__(self, cfg: DictConfig, class_weights: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model_cfg = cfg.model
        self.training_cfg = cfg.training
        self.automatic_optimization = False

        self.model = self._build_model()

        alpha = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )
        self.criterion = FocalLoss(gamma=cfg.training.focal_gamma, alpha=alpha)

        self._current_stage = "head"
        self._stage_epoch_count = 0

        self._val_preds: list = []
        self._val_labels: list = []

    def _build_model(self) -> nn.Module:
        cfg = self.model_cfg
        if cfg.name == "simple_cnn":
            return SimpleCNN(num_classes=cfg.num_classes, dropout=cfg.dropout)
        elif cfg.name == "efficientnet":
            model = SkinLesionEfficientNet(
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                dropout=cfg.dropout,
            )
            if cfg.freeze_backbone:
                model.freeze_backbone()
            return model
        else:
            raise ValueError(f"Неизвестная модель: {cfg.name}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray) -> dict:
        num_classes = self.model_cfg.num_classes
        tp = np.zeros(num_classes)
        fp = np.zeros(num_classes)
        fn = np.zeros(num_classes)
        support = np.zeros(num_classes)

        for cls in range(num_classes):
            tp[cls] = ((preds == cls) & (labels == cls)).sum()
            fp[cls] = ((preds == cls) & (labels != cls)).sum()
            fn[cls] = ((preds != cls) & (labels == cls)).sum()
            support[cls] = (labels == cls).sum()

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return {
            "accuracy": float((preds == labels).mean()),
            "balanced_accuracy": float((tp / (support + 1e-10)).mean()),
            "macro_f1": float(f1.mean()),
            "macro_recall": float(recall.mean()),
            "mel_recall": float(recall[CLASS_NAMES.index("mel")]),
            "nv_recall": float(recall[CLASS_NAMES.index("nv")]),
        }

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        opt = self.optimizers()
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        opt.step()

        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        self._val_preds.append(preds.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        all_preds = np.concatenate(self._val_preds)
        all_labels = np.concatenate(self._val_labels)
        metrics = self._compute_metrics(all_preds, all_labels)

        bal_acc = metrics["balanced_accuracy"]

        self.log("val/accuracy", metrics["accuracy"], prog_bar=True, sync_dist=True)
        self.log("val/balanced_accuracy", bal_acc, prog_bar=True, sync_dist=True)
        self.log("val/macro_f1", metrics["macro_f1"], sync_dist=True)
        self.log("val/mel_recall", metrics["mel_recall"], sync_dist=True)
        self.log("val/nv_recall", metrics["nv_recall"], sync_dist=True)

        if bal_acc > getattr(self, "_best_bal_acc", 0.0):
            self._best_bal_acc = bal_acc
            ckpt_path = (
                f"checkpoints/manual_best_epoch={self.current_epoch}_bal_acc={bal_acc:.4f}.ckpt"
            )
            self.trainer.save_checkpoint(ckpt_path)
            print(f"\n[BEST] Новый лучший: {bal_acc:.4f} → {ckpt_path}")

        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch
        outputs = self(images)
        preds = outputs.argmax(dim=1)

        self._val_preds.append(preds.cpu().numpy())
        self._val_labels.append(labels.cpu().numpy())

    def on_test_epoch_end(self) -> None:
        all_preds = np.concatenate(self._val_preds)
        all_labels = np.concatenate(self._val_labels)
        metrics = self._compute_metrics(all_preds, all_labels)

        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ")
        print("=" * 60)
        for key, value in metrics.items():
            print(f"  {key:<25} {value:.4f}")
            self.log(f"test/{key}", value, sync_dist=True)

        self._val_preds.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        """Начальный оптимизатор (этап 1: только голова)."""
        return torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.parameters()),
            lr=self.training_cfg.lr_head,
            weight_decay=self.training_cfg.weight_decay,
        )

    def on_train_epoch_start(self) -> None:
        """Переключение стадий разморозки."""
        current_epoch = self.current_epoch

        if not isinstance(self.model, SkinLesionEfficientNet):
            return

        if current_epoch == self.training_cfg.epochs_head:
            print("\n[STAGE 2] Разморозка последних блоков")
            self.model.unfreeze_from(self.model_cfg.unfreeze_from)
            self._update_optimizer(
                lr_backbone=self.training_cfg.lr_finetune / 10,
                lr_head=self.training_cfg.lr_finetune,
            )

        elif current_epoch == self.training_cfg.epochs_head + self.training_cfg.epochs_finetune:
            print("\n[STAGE 3] Полная разморозка")
            self.model.unfreeze_all()
            self._update_optimizer(
                lr_backbone=self.training_cfg.lr_full_backbone,
                lr_head=self.training_cfg.lr_full_head,
            )

    def _update_optimizer(self, lr_backbone: float, lr_head: float) -> None:
        """Обновляет оптимизатор для новой стадии."""
        param_groups = self.model.get_param_groups(lr_backbone, lr_head)
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.training_cfg.weight_decay,
        )
        self.optimizers().__class__ = optimizer.__class__
        self.optimizers().__dict__.update(optimizer.__dict__)

    def export_onnx(self, output_path: str, image_size: int = 224) -> None:
        self.model.eval()
        device = next(self.model.parameters()).device
        dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path_obj),
            export_params=True,
            opset_version=17,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
        print(f"[ONNX] Модель экспортирована: {output_path_obj}")
