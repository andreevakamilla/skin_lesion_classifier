"""Глобальные константы проекта."""

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

CLASS_FULL_NAMES = {
    "akiec": "Актинический кератоз",
    "bcc": "Базальноклеточная карцинома",
    "bkl": "Доброкачественный кератоз",
    "df": "Дерматофиброма",
    "mel": "Меланома",
    "nv": "Меланоцитарный невус",
    "vasc": "Сосудистое поражение",
}

MALIGNANT_CLASSES = {"mel", "bcc", "akiec"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
