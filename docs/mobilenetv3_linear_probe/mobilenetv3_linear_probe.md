# 实验记录：MobileNetV3 线性探测 (Linear Probing)

## 1. 实验配置
*   **模型架构**：MobileNetV3 Large
*   **训练策略**：线性探测 (Linear Probing)，冻结主干网络 (`freeze_backbone: true`)
*   **输入尺寸**：224 x 224
*   **Batch Size**：64
*   **优化器**：AdamW
*   **学习率**：0.0001 (初始)，使用余弦退火 (Cosine Annealing) 衰减
*   **训练轮数**：30 Epochs

## 2. 核心结果数据
*   **最佳验证集准确率 (Best val_acc)**：**67.59%** (诞生于 Epoch 30)
*   **最佳验证集 F1 分数 (Best val_f1)**：**0.5615** (诞生于 Epoch 30)
*   **最终训练损失 (Final train_loss)**：0.8037
*   **模型权重路径**：`experiments/checkpoints/best_model.pth`

## 3. 报告可用话术（直接发给成员 C）
> “与 ResNet18 相比，轻量级模型 MobileNetV3 Large 在线性探测策略下表现出了相近的学习能力。经过 30 个 Epoch，其最终验证集准确率达到 **67.59%**，F1-score 为 **0.5615**。值得注意的是，MobileNetV3 在最后几轮依然有轻微的上升趋势，未出现明显的过拟合，说明其泛化能力较强，但在绝对精度上略低于 ResNet18 (68.80%)。这符合两者参数量与特征提取深度的差异。”

## 4. 附件与可视化资产
*   **终端日志**：见同级目录下的 `mobilenetv3_linear_probe.txt`
*   **混淆矩阵图**：`experiments/confusion_matrix.png` (已被本次最佳结果覆盖)
*   **TensorBoard 曲线**：包含与 ResNet18 的对比曲线截图。