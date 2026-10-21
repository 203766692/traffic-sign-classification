# 实验记录：ResNet18 微调 (Fine-tuning)

## 1. 实验配置
*   **模型架构**：ResNet18
*   **训练策略**：微调 (Fine-tuning)，解冻全部网络层 (`freeze_backbone: false`)
*   **输入尺寸**：224 x 224
*   **Batch Size**：32 (因显存限制调小)
*   **优化器**：AdamW
*   **学习率**：0.00001 (初始，小学习率防止灾难性遗忘)，余弦退火衰减
*   **训练轮数**：10 Epochs

## 2. 核心结果数据
*   **最佳验证集准确率 (Best val_acc)**：**99.19%** (诞生于 Epoch 9)
*   **最佳验证集 F1 分数 (Best val_f1)**：**0.9894** (诞生于 Epoch 9)
*   **最终训练损失 (Final train_loss)**：0.0041
*   **模型权重路径**：`experiments/checkpoints/best_model.pth`

## 3. 报告可用话术（直接发给成员 C）
> “相比于线性探测（68.80%），ResNet18 在解冻全网进行微调后，展现出了极其惊人的性能飞跃。在仅训练 10 个 Epoch 的情况下，验证集准确率飙升至 **99.19%**，F1-score 达到 **0.9894**。这充分证明了在大规模数据集上预训练的视觉模型（ImageNet）拥有强大的特征泛化能力，只需极小的学习率（1e-5）进行微调，即可迅速适配到全新的 GTSRB 交通标志领域，且未出现明显的过拟合现象。”

## 4. 附件与可视化资产
*   **终端日志**：见同级目录下的 `resnet18_finetune.txt`
*   **混淆矩阵图**：`experiments/confusion_matrix.png` (近乎完美的对角线)
*   **TensorBoard 曲线**：包含 Loss 与 Accuracy 断崖式优化曲线。