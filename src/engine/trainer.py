"""PyTorch 训练与验证循环。"""

from typing import Any, Dict, Optional, Union

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.utils.metrics import calculate_accuracy, calculate_f1_score


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    """未显式指定设备时，自动优先使用 CUDA，否则回退到 CPU。"""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """完成一个 epoch 的训练，并返回平均损失与准确率。"""

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.train()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in dataloader:
        images = images.to(resolved_device)
        labels = labels.to(resolved_device)

        # 标准训练步骤：清空梯度、前向传播、反向传播、更新参数。
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    batch_count = max(len(dataloader), 1)
    average_loss = running_loss / batch_count
    accuracy = calculate_accuracy(all_labels, all_predictions)
    return {"loss": average_loss, "accuracy": accuracy}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """在验证集上计算损失、准确率、F1-score 和预测结果。"""

    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    model.eval()

    running_loss = 0.0
    all_labels = []
    all_predictions = []

    # 验证阶段只做前向推理，不计算梯度。
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(resolved_device)
            labels = labels.to(resolved_device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            all_labels.extend(labels.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

    batch_count = max(len(dataloader), 1)
    average_loss = running_loss / batch_count
    return {
        "loss": average_loss,
        "accuracy": calculate_accuracy(all_labels, all_predictions),
        "f1": calculate_f1_score(all_labels, all_predictions),
        "y_true": all_labels,
        "y_pred": all_predictions,
    }
