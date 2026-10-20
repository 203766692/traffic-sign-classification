"""分类任务评估指标与混淆矩阵可视化工具。"""

from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def _to_numpy(values: Any) -> np.ndarray:
    """将 PyTorch 张量、列表等输入统一转换为 NumPy 数组。"""

    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def calculate_accuracy(y_true: Any, y_pred: Any) -> float:
    """计算分类准确率。"""

    return float(accuracy_score(_to_numpy(y_true), _to_numpy(y_pred)))


def calculate_f1_score(
    y_true: Any,
    y_pred: Any,
    average: str = "macro",
) -> float:
    """计算 F1-score，默认使用更适合多分类任务的宏平均。"""

    return float(
        f1_score(
            _to_numpy(y_true),
            _to_numpy(y_pred),
            average=average,
            zero_division=0,
        )
    )


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    save_path: str = "experiments/confusion_matrix.png",
    class_names: Optional[Sequence[str]] = None,
) -> Path:
    """计算并保存混淆矩阵，返回图片路径。"""

    true_values = _to_numpy(y_true)
    predicted_values = _to_numpy(y_pred)
    matrix = confusion_matrix(true_values, predicted_values)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=False,
        cmap="Blues",
        xticklabels=class_names if class_names is not None else True,
        yticklabels=class_names if class_names is not None else True,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
