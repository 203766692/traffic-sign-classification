# 实验记录：ResNet18 线性探测 (Linear Probing)

## 1. 实验配置
*   **模型架构**：ResNet18
*   **训练策略**：线性探测 (Linear Probing)，冻结主干网络 (`freeze_backbone: true`)
*   **输入尺寸**：224 x 224
*   **Batch Size**：64
*   **优化器**：AdamW
*   **学习率**：0.0001 (初始)，使用余弦退火 (Cosine Annealing) 衰减
*   **训练轮数**：30 Epochs

## 2. 核心结果数据
*   **最佳验证集准确率 (Best val_acc)**：**68.80%** (诞生于 Epoch 25)
*   **最佳验证集 F1 分数 (Best val_f1)**：**0.5802** (诞生于 Epoch 25)
*   **最终训练损失 (Final train_loss)**：0.8525
*   **模型权重路径**：`experiments/checkpoints/best_model.pth`

## 3. 报告可用话术（直接发给成员 C）
> “经过 30 个 Epoch 的余弦退火训练，采用线性探测策略的 ResNet18 最终达到 **68.8%** 的验证集准确率，F1-score 达到 **0.58**。模型在 Epoch 25 达到最佳状态，随后由于学习率衰减至 0，出现轻微过拟合现象（验证集 Loss 略升），但代码通过最佳模型保存机制锁定了最优权重。”

## 4. 附件与可视化资产
*   **终端日志**：见同级目录下的 `resnet18_linear_probe.txt`
*   **混淆矩阵图**：`experiments/confusion_matrix.png` (展示具体哪些类别容易混淆)
*   **TensorBoard 曲线**：包含 Loss 和 Accuracy 曲线截图 (随报告提交)