"""基于 torchvision 的迁移学习模型构建接口。"""

from torch import nn
from torchvision import models


def build_model(
    model_name: str,
    num_classes: int,
    freeze_backbone: bool,
    pretrained: bool = True,
) -> nn.Module:
    """构建并返回指定模型，同时完成分类层替换和特征层冻结。"""

    normalized_name = model_name.lower()

    if normalized_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

        # 线性探测时冻结 ResNet 的全部原始特征参数。
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        # 替换后的全连接层是新参数，默认保持可训练。
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if normalized_name == "mobilenet_v3":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)

        # 线性探测时冻结 MobileNetV3 的全部原始特征参数。
        if freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad = False

        # MobileNetV3 的最后一层位于 classifier 序列的末尾。
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    supported_models = "resnet18、mobilenet_v3"
    raise ValueError(
        f"不支持的模型名称：{model_name}。当前支持：{supported_models}"
    )
