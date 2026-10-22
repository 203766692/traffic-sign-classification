# 实验记录：MobileNetV3 微调 (Fine-tuning)

## 1. 实验配置
*   **模型架构**：MobileNetV3 Large
*   **训练策略**：微调 (Fine-tuning)，解冻全部网络层 (`freeze_backbone: false`)
*   **输入尺寸**：224 x 224
*   **Batch Size**：32
*   **优化器**：AdamW
*   **学习率**：0.00001 (初始)，余弦退火衰减
*   **训练轮数**：10 Epochs

## 2. 核心结果数据
*   **最佳验证集准确率 (Best val_acc)**：**96.63%** (诞生于 Epoch 10)
*   **最佳验证集 F1 分数 (Best val_f1)**：**0.9431** (诞生于 Epoch 10)
*   **最终训练损失 (Final train_loss)**：0.0285
*   **模型权重路径**：`experiments/checkpoints/best_model.pth`

## 3. 报告可用话术（直接发给成员 C）
> “作为轻量级网络的代表，MobileNetV3 在采用微调策略后，性能同样实现了质的飞跃。经过 10 个 Epoch 的训练，其验证集准确率达到了 **96.63%**，F1-score 为 **0.9431**。尽管在绝对精度上略低于重量级的 ResNet18（99.2%），但考虑到 MobileNetV3 在参数量和推理速度上的巨大优势，其在车载边缘计算、实时交通标志识别等场景中展现出了极高的实用价值与性价比。”

## 4. 附件与可视化资产
*   **终端日志**：同级目录下的 `mobilenetv3_finetune.txt`
*   **混淆矩阵图**：`confusion_matrix.png` (当前已更新为 MobileNetV3 微调版)
*   **TensorBoard 曲线**：包含微调过程的曲线截图。