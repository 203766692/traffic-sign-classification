"""GTSRB 交通标志分类训练入口。"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    StepLR,
)
from torch.utils.tensorboard import SummaryWriter

# 保证从项目根目录或 scripts 目录执行时都能导入 src 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.gtsrb import get_dataloaders
from src.engine.trainer import evaluate, train_one_epoch
from src.models.builder import build_model
from src.utils.metrics import plot_confusion_matrix
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 GTSRB 分类模型")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="YAML 配置文件路径",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """读取 YAML 配置，并把相对路径解析为项目根目录下的绝对路径。"""

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"配置文件内容无效：{path}")

    data_root = Path(config["data_root"])
    if not data_root.is_absolute():
        config["data_root"] = str(PROJECT_ROOT / data_root)

    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    config["output_dir"] = str(output_dir)
    return config


def build_scheduler(
    optimizer: Optimizer,
    config: dict,
    num_epochs: int,
) -> Optional[LRScheduler]:
    """根据配置构建学习率调度器。"""

    scheduler_name = str(config.get("lr_scheduler", "cosine")).lower()
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    if scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=int(config.get("lr_step_size", 10)),
            gamma=float(config.get("lr_gamma", 0.1)),
        )
    if scheduler_name in {"none", "disabled"}:
        return None
    raise ValueError(
        f"不支持的调度器：{scheduler_name}，可选值为 cosine、step 或 none"
    )


def main() -> None:
    """读取配置并执行完整训练流程。"""

    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    model = build_model(
        model_name=str(config["model_name"]),
        num_classes=int(config["num_classes"]),
        freeze_backbone=bool(config["freeze_backbone"]),
        pretrained=bool(config.get("pretrained", True)),
    )
    train_loader, val_loader = get_dataloaders(config)

    criterion = nn.CrossEntropyLoss()
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )

    num_epochs = int(config["num_epochs"])
    scheduler = build_scheduler(optimizer, config, num_epochs)
    output_dir = Path(config["output_dir"])
    checkpoint_path = output_dir / "checkpoints" / "best_model.pth"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_accuracy = -1.0
    try:
        for epoch in range(1, num_epochs + 1):
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            # 每个 epoch 结束后更新学习率。
            if scheduler is not None:
                scheduler.step()

            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"train_acc={train_metrics['accuracy']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"lr={learning_rate:.6f}"
            )

            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            writer.add_scalar("LearningRate", learning_rate, epoch)

            # 仅当验证集准确率刷新纪录时覆盖保存最佳模型。
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                torch.save(model.state_dict(), checkpoint_path)
                plot_confusion_matrix(
                    val_metrics["y_true"],
                    val_metrics["y_pred"],
                    save_path=str(output_dir / "confusion_matrix.png"),
                )
                print(
                    f"已保存最佳模型：val_acc={best_accuracy:.4f}，"
                    f"路径={checkpoint_path}"
                )
    finally:
        writer.close()


if __name__ == "__main__":
    main()
