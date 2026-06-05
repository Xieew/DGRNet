# Improved RegNet with Dual Attention for English Proficiency Assessment

> 官方 PyTorch 实现 | 论文发表于 *Information* 期刊，标题：*English Proficiency Assessment Based on Improved RegNet and Dual Attention Mechanism*

本仓库提供了基于改进 RegNet + 可变形注意力 (DAT) + 图注意力 (GAT) 的英语学业水平评估模型，实现英语专业学生“达标/预警”二分类，支持智慧校园中的智能化教学管理。

## 主要特性

- **双卷积级联 Stem**：扩大浅层感受野，增强对成绩波动、出勤模式等学术特征的捕获。
- **可变形注意力 (DAT)**：动态聚焦 GAF 图像中的关键区域，精准定位学业下滑前期的细微纹理变化。
- **图注意力网络 (GAT)**：建模听力、阅读、口语、作业等课程指标之间的相互影响关系。
- **端到端训练**：将 GAF 转换后的二维图像直接输入模型，输出预警概率。

##  项目文件结构
```text
DGRNet/
|-- Ac/
|   |-- train/
|   |   |-- Eligible/
|   |   `-- NotEligible/
|   |-- val/
|   |   |-- Eligible/
|   |   `-- NotEligible/
|   `-- test/
|       |-- Eligible/
|       `-- NotEligible/
|-- weights3_fast/
|-- model3.py
|-- train3.py
|-- my_dataset.py
|-- utils.py
|-- training_fast.txt
`-- README.md
```

## 环境配置
### 依赖安装（Python 3.10）
```bash
conda create -n regnet-dual python=3.10
conda activate regnet-dual

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy matplotlib opencv-python pandas pillow tensorboard tqdm timm einops
```

硬件要求
GPU：建议显存 ≥ 6GB（如 RTX 3060/4060），训练 100 轮约 2-3 小时。

CPU：仅推理可用（15~20 ms/张），不建议训练。

## 1. 准备数据集
将 GAF 转换后的图像按上述 Ac/ 结构放置（参考论文第 2 节生成 GAF 图像，或使用 pyts.image.GramianAngularField）。数据集划分比例建议为 3:1:1（训练:验证:测试）。
## 2. 训练模型
确保已激活虚拟环境，然后运行：
