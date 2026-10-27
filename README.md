# 基于迁移学习的交通标志图像分类与模型对比研究

本项目为《多媒体技术》课程实验项目。旨在基于 GTSRB 数据集，利用迁移学习（线性探测与微调）对比不同网络架构（如 ResNet18、MobileNetV3 等）在图像分类任务上的精度与效率。

## 📂 项目结构说明

```text
traffic-sign-classification/
├── configs/                  # 配置文件目录
│   └── config.yaml           # 全局配置（数据路径、超参数、模型选择等）
├── data/                     # 数据集目录（已被 .gitignore 忽略）
│   └── gtsrb/                # GTSRB 数据集解压后的存放路径
├── experiments/              # 实验产物目录（已被 .gitignore 忽略）
│   ├── logs/                 # TensorBoard 训练日志
│   └── checkpoints/          # 训练过程中的模型权重保存目录 (*.pth)
├── notebooks/                # 探索性分析 (EDA) 与可视化 Jupyter Notebooks
├── scripts/                  # 可执行入口脚本
│   └── train.py              # 训练主脚本（解析参数、读取配置、启动训练）
├── src/                      # 核心源代码模块
│   ├── datasets/             # 数据加载与预处理
│   │   └── gtsrb.py          # 自定义 Dataset 类、数据增强 (Transforms) 和 DataLoader 构建
│   ├── engine/               # 训练与评估引擎
│   │   └── trainer.py        # 包含训练循环、验证循环、早停、模型保存等核心逻辑
│   ├── models/               # 模型定义
│   │   └── builder.py        # 模型构建器（加载预训练权重、替换 43 类分类头、冻结/微调策略）
│   └── utils/                # 通用工具函数
│       ├── metrics.py        # 评价指标计算（Accuracy, F1-score, 混淆矩阵等）
│       ├── seed.py           # 设置全局随机种子（保证实验可复现性）
│       └── visualize.py      # 可视化工具（Grad-CAM, t-SNE 等，预留）
├── .gitignore                # Git 忽略文件配置（忽略数据集、虚拟环境、模型权重等）
├── requirements.txt          # 项目依赖包列表
└── README.md                 # 项目说明文档（即本文件）
```

## 📊 实验结果概览

本项目基于 GTSRB 数据集，全面完成了四大核心实验的闭环，实验记录与可视化图表已归档至 `experiments/logs/` 目录。

| 模型架构 | 训练策略 | 实验状态 | 备注说明 |
| :--- | :--- | :--- | :--- |
| ResNet18 | 线性探测 (Linear Probe) | ✅ 已完成 | Loss 与学习率衰减曲线已归档 |
| ResNet18 | 微调 (Fine-tuning) | ✅ 已完成 | **准确率突破 99%**，表现出色 |
| MobileNetV3 | 线性探测 (Linear Probe) | ✅ 已完成 | 实验记录与数据可视化已整理 |
| MobileNetV3 | 微调 (Fine-tuning) | ✅ 已完成 | 实验结果已闭环归档 |

*(注：详细的超参数配置、Loss/Acc 曲线与深度对比分析，请参见课程实验报告或 `experiments/` 目录下的可视化图表。)*

## 💻 环境准备

请确保已安装 Python 3.8+，然后执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 🚀 训练机操作指南（针对有 GPU 的机器）

### 1. 准备数据集 (GTSRB)

本项目使用德国交通标志识别基准 (GTSRB) 数据集。鉴于原官网（benchmark.ini.rub.de）在国内网络环境下经常无法访问或下载中断，推荐直接使用 Kaggle 镜像源获取。

**镜像地址**：
[https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

请登录训练机，在项目根目录下根据网络环境选择以下任意一种方式：

**方式一：使用 kagglehub 库（最简洁，无需配置 API 令牌）**
```python
import kagglehub
# 自动下载并缓存到本地
path = kagglehub.dataset_download("meowmeowmeowmeowmeow/gtsrb-german-traffic-sign")
print("数据集路径:", path)
```

**方式二：使用 Kaggle CLI（需配置 API）**
1. 在 [Kaggle 账户设置](https://www.kaggle.com/account) 中点击 "Create New Token" 下载 `kaggle.json`。
2. 将文件上传至训练机的 `~/.kaggle/` 目录（Linux/macOS）或 `C:\Users\<用户名>\.kaggle\`（Windows），并确保权限正确。
3. 在项目根目录执行：
```bash
# 下载数据集
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
# 解压至 data/gtsrb 目录
unzip gtsrb-german-traffic-sign.zip -d data/gtsrb/
```

**方式三：使用 wget 直接下载**
```bash
mkdir -p data/gtsrb
wget https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign/download/archive.zip -O data/gtsrb/archive.zip
unzip data/gtsrb/archive.zip -d data/gtsrb/
```

**解压后的目录结构参考**（Kaggle 镜像包内通常包含 `Train/`、`Test/` 文件夹及对应的 CSV 文件）：
```text
data/gtsrb/
├── Train/
├── Test/
├── Meta.csv
├── Train.csv
└── Test.csv
```

### 2. 启动训练
确保代码已同步至训练机，且数据集已就位。在项目根目录执行：
```bash
# 示例：使用 ResNet18 进行微调训练
python scripts/train.py --config configs/config.yaml
```
*(注：项目代码已全面开发完毕，包含数据加载、模型构建、训练引擎等完整模块，支持一键启动训练与评估。)*

## 👥 协作说明

- `main` 分支为保护分支，禁止直接提交。
- 开发新功能请切出 `feature/xxx` 分支，完成后提交 Pull Request。
- 提交前请确保未将 `data/` 目录或模型权重文件（`*.pth`）加入暂存区。
