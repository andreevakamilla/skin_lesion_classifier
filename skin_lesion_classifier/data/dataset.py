from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from skin_lesion_classifier.constants import CLASS_TO_IDX, IMAGENET_MEAN, IMAGENET_STD


def download_data(data_dir: str = "data") -> None:
    import urllib.request

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    metadata_url = (
        "https://dataverse.harvard.edu/api/access/datafile/" "3172582?format=original&gbrecs=true"
    )
    metadata_dest = data_path / "HAM10000_metadata.csv"

    if not metadata_dest.exists():
        print(f"Скачиваем метаданные в {metadata_dest}...")
        urllib.request.urlretrieve(metadata_url, metadata_dest)
        print("Готово.")
    else:
        print(f"Метаданные уже есть: {metadata_dest}")


def get_train_transforms(image_size: int, cfg: DictConfig) -> A.Compose:
    """Аугментации для обучения."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=cfg.horizontal_flip_prob),
            A.VerticalFlip(p=cfg.vertical_flip_prob),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=cfg.rotate_limit,
                border_mode=0,
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=cfg.brightness_limit,
                        contrast_limit=cfg.contrast_limit,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=cfg.hue_shift_limit,
                        p=1.0,
                    ),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                num_holes_range=(1, cfg.dropout_num_holes),
                hole_height_range=(8, cfg.dropout_max_size),
                hole_width_range=(8, cfg.dropout_max_size),
                fill=0,
                p=0.3,
            ),
            A.GaussNoise(std_range=(0.04, 0.2), p=cfg.gauss_noise_prob),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int) -> A.Compose:
    """Аугментации для валидации и теста."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class HAM10000Dataset(Dataset):
    """PyTorch Dataset для HAM10000."""

    def __init__(
        self,
        image_paths: list[str],
        labels: list[int],
        transform: Optional[A.Compose] = None,
    ) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image_array = np.array(image)

        if self.transform:
            image_array = self.transform(image=image_array)["image"]

        return image_array, self.labels[idx]


def find_image_path(image_id: str, data_dir: Path) -> Optional[str]:
    """Ищет изображение в подпапках data_dir."""
    for ext in [".jpg", ".png"]:
        candidates = [
            data_dir / f"{image_id}{ext}",
            data_dir / "HAM10000_images_part_1" / f"{image_id}{ext}",
            data_dir / "HAM10000_images_part_2" / f"{image_id}{ext}",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
    return None


def load_metadata(metadata_path: Path, data_dir: Path) -> pd.DataFrame:
    """Загружает метаданные и добавляет пути к изображениям."""
    dataframe = pd.read_csv(metadata_path)
    dataframe["image_path"] = dataframe["image_id"].apply(
        lambda image_id: find_image_path(image_id, data_dir)
    )

    missing_count = dataframe["image_path"].isna().sum()
    if missing_count > 0:
        print(f"[WARNING] Не найдено {missing_count} изображений из {len(dataframe)}")
        dataframe = dataframe.dropna(subset=["image_path"]).reset_index(drop=True)

    dataframe["label"] = dataframe["dx"].map(CLASS_TO_IDX)
    return dataframe


def stratified_group_split(
    dataframe: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified Group Split по lesion_id — без утечки данных."""
    labels = dataframe["label"].values
    groups = dataframe["lesion_id"].values

    splitter_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    train_val_idx, test_idx = next(splitter_test.split(dataframe, labels, groups))

    df_train_val = dataframe.iloc[train_val_idx]
    val_relative = val_ratio / (train_ratio + val_ratio)

    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_relative, random_state=random_seed)
    labels_tv = df_train_val["label"].values
    groups_tv = df_train_val["lesion_id"].values
    train_idx_rel, val_idx_rel = next(splitter_val.split(df_train_val, labels_tv, groups_tv))

    df_train = dataframe.iloc[train_val_idx[train_idx_rel]].reset_index(drop=True)
    df_val = dataframe.iloc[train_val_idx[val_idx_rel]].reset_index(drop=True)
    df_test = dataframe.iloc[test_idx].reset_index(drop=True)

    train_lesions = set(df_train["lesion_id"])
    assert not (train_lesions & set(df_val["lesion_id"])), "Утечка train-val!"
    assert not (train_lesions & set(df_test["lesion_id"])), "Утечка train-test!"

    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)} | Утечек нет ✓")
    return df_train, df_val, df_test


def compute_class_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Вычисляет веса классов (inverse frequency)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    return weights / weights.sum() * num_classes


class HAM10000DataModule(pl.LightningDataModule):
    """Lightning DataModule для HAM10000."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.df_train: Optional[pd.DataFrame] = None
        self.df_val: Optional[pd.DataFrame] = None
        self.df_test: Optional[pd.DataFrame] = None

    def setup(self, stage: Optional[str] = None) -> None:
        dataframe = load_metadata(
            Path(self.cfg.metadata_file),
            Path(self.cfg.data_dir),
        )
        self.df_train, self.df_val, self.df_test = stratified_group_split(
            dataframe,
            train_ratio=self.cfg.train_ratio,
            val_ratio=self.cfg.val_ratio,
            test_ratio=self.cfg.test_ratio,
            random_seed=self.cfg.random_seed,
        )

    def _make_loader(self, dataframe: pd.DataFrame, is_train: bool = False) -> DataLoader:
        transform = (
            get_train_transforms(self.cfg.image_size, self.cfg.augmentation)
            if is_train
            else get_val_transforms(self.cfg.image_size)
        )
        dataset = HAM10000Dataset(
            image_paths=dataframe["image_path"].tolist(),
            labels=dataframe["label"].tolist(),
            transform=transform,
        )

        sampler = None
        shuffle = is_train
        if is_train and self.cfg.use_weighted_sampler:
            weights = compute_class_weights(dataframe["label"].values, len(CLASS_TO_IDX))
            sample_weights = weights[dataframe["label"].values]
            sampler = WeightedRandomSampler(sample_weights, len(dataframe), replacement=True)
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.cfg.get("batch_size", 32),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=is_train,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.df_train, is_train=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.df_val, is_train=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.df_test, is_train=False)
