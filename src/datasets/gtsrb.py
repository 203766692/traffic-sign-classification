"""GTSRB 数据集读取与 DataLoader 构建。"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class GTSRBDataset(Dataset):
    """基于 GTSRB CSV 标注文件构建的 PyTorch 数据集。"""

    def __init__(
        self,
        csv_file: str,
        data_root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.csv_file = Path(csv_file)
        self.transform = transform

        if not self.csv_file.is_file():
            raise FileNotFoundError(f"找不到 GTSRB 标注文件：{self.csv_file}")

        # CSV 中既包含标签，也包含每张图片相对于数据根目录的路径。
        self.annotations = pd.read_csv(self.csv_file)
        required_columns = {"Path", "ClassId"}
        missing_columns = required_columns.difference(self.annotations.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"GTSRB 标注文件缺少必要字段：{missing}")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        row = self.annotations.iloc[index]
        image_path = Path(str(row["Path"]))

        # 某些 CSV 使用绝对路径，另一些使用相对于 data_root 的路径。
        if not image_path.is_absolute():
            image_path = self.data_root / image_path

        if not image_path.is_file():
            raise FileNotFoundError(f"找不到图像文件：{image_path}")

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, int(row["ClassId"])


def _build_transforms(image_size: int) -> Tuple[Callable, Callable]:
    """构建训练集增强流程和验证集预处理流程。"""

    # GTSRB 图像尺寸不一，先统一缩放后再进行随机裁剪与色彩扰动。
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop((image_size, image_size), padding=4),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    # 验证阶段不做随机增强，只保留确定性的缩放和归一化。
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return train_transform, val_transform


def get_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """根据配置返回训练集和验证集 DataLoader。"""

    data_root = Path(config["data_root"])
    train_csv = data_root / "Train.csv"
    val_csv = data_root / "Test.csv"
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))

    train_transform, val_transform = _build_transforms(image_size)
    train_dataset = GTSRBDataset(
        csv_file=str(train_csv),
        data_root=str(data_root),
        transform=train_transform,
    )
    val_dataset = GTSRBDataset(
        csv_file=str(val_csv),
        data_root=str(data_root),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, val_loader
