# Improved RegNet with Dual Attention for English Proficiency Assessment

官方 PyTorch 实现 | 论文发表于 *Information* 期刊，标题：《English Proficiency Assessment Based on Improved RegNet and Dual Attention Mechanism》

提出改进 RegNet 架构融合可变形注意力（DAT）与图注意力（GAT）的模型，实现英语专业学生学业达标/预警的二分类，辅助智慧校园中的精准教学管理。

## 1. 研究背景与模型定位

英语专业招生规模持续扩大，传统基于统计分析和规则引擎的学业评估方法在效率与精度上存在明显不足，难以支撑个性化教学与早期预警。现有深度学习方法在处理一维时序成绩数据时，面临长期依赖建模困难、多源异构特征融合不充分、注意力机制难以平衡局部与全局表征等问题。

本文提出基于改进 RegNet 与双注意力融合的英语水平评估模型。首先通过 Gramian Angular Field（GAF）将一维成绩序列编码为二维图像，保留时序依赖关系；然后优化 RegNet 的初始下采样模块（双卷积级联），扩大感受野以增强浅层学术特征提取；最后引入可变形注意力（DAT）聚焦关键学业区域，并结合图注意力网络（GAT）建模多课程指标间的深层关联。该模型在自建英语专业学业数据集上达到 **99.46%** 的平均准确率，为智慧校园背景下的学业智能管理提供了有效技术方案。
## 2. 改进 RegNet 核心创新点

- **双卷积级联下采样模块（Stem 优化）**  
  将原始 RegNet 的单层卷积替换为两个 `3×3` 卷积（第一个 stride=2 降采样，第二个 stride=1 保持分辨率）。在几乎不增加参数量的前提下显著扩大感受野，增强对学期成绩波动、出勤模式等浅层学术特征的捕获能力。

- **可变形注意力机制（Deformable Attention, DAT）**  
  在 RegNet 的主干阶段嵌入 DAT，通过可学习偏移量动态选择信息最丰富的空间区域。能够精准定位成绩骤降前期的细微纹理变化、异常出勤等不规则预警区域，有效补偿下采样导致的空间细节损失。

- **图注意力网络（Graph Attention Network, GAT）**  
  在相同阶段并行引入 GAT，将不同课程指标（听力、阅读、口语、作业等）构建为图结构，利用多头注意力自动学习节点间的相互影响权重。例如当听力成绩下滑时，模型会增强对口语和视听说课程的关注，抑制单一课程异常带来的误判。

- **三层渐进式特征融合架构**  
  改进的 RegNet 提取宏观性能趋势与微观波动细节；DAT 自适应采样不规则预警区域；GAT 显式建模课程间关联强度。三者实现端到端联合优化，同时完成空间定位、局部聚焦与关系建模。
## 3. 实验数据集：English Academic Dataset

### 3.1 数据集概况

本研究使用的数据集来源于某高校英语专业本科生的四年真实学业记录，原始数据包括：课堂出勤率、线上参与度、语言实践完成情况，以及听、说、读、写四项技能的成绩。通过 GAF 将每位学生的时序成绩序列转换为二维图像，构建分类任务（达标 / 未达标）。

| 数据集名称 | 包含类别 | 图像总数 | 图像分辨率 | 数据分布（训练:验证:测试） |
| :---: | :---: | :---: | :---: | :---: |
| English Academic Dataset | 达标 / 未达标 | 6165 | 224×224 | 3:1:1 |

### 3.2 数据集结构

仓库中 `Ac/` 文件夹组织如下（需自行通过 GAF 转换生成）：
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
## 4. 实验环境配置

### 4.1 依赖安装

推荐使用 Anaconda 创建虚拟环境，Python 3.10，PyTorch 2.0.1（已验证兼容）：

```bash
# 1. 创建并激活虚拟环境
conda create -n regnet-dual python=3.10
conda activate regnet-dual

# 2. 安装 PyTorch（CUDA 11.8 示例，CPU 用户可替换为 cpu 版本）
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# 3. 安装其他依赖库
pip install timm==1.0.15 einops==0.8.1 tensorboard==2.10.0 tqdm pillow numpy
```

### 4.2 硬件要求
GPU：推荐 NVIDIA GPU（显存 ≥ 6GB，如 RTX 3060/4060），训练 100 轮约 2-3 小时，显存占用峰值 ≤ 5GB。

CPU：支持推理测试（单张图像约 15 ms），但不推荐用于完整训练流程。

## 5. 实验结果

### 5.1 核心指标对比（Academic Dataset）
本文方法（DGRNet = Improved RegNet + DAT + GAT）与多种主流模型在英语学业达标预测任务上的性能对比如下：

```bash
模型	分类准确率（Accuracy）	计算量（FLOPs）	参数量（M）
ResNet-50	90.66%	4.1 G	25.6
EfficientNetV2	91.32%	2.9 G	20.1
ViT-Base	94.20%	17.6 G	86.8
ConvNeXt	93.54%	8.7 G	27.8
DGRNet（本文）	99.46%	12.84 G	68.3
注：1. 准确率来自论文表4，多次实验平均值；2. 单独添加 DAT 使准确率从 90.66% 提升至 94.30%（+3.64%），单独添加 GAT 提升至 94.18%（+3.52%），两者联合达到 99.46%；3. 推理速度为 15.84 ms/张（RTX 4060）。
```

## 6. 代码使用说明

### 6.1 模型训练
运行 train3.py 脚本启动训练，支持通过参数调整训练配置，示例命令（适配 Academic Dataset）：

```bash
python train3.py \
  --data-path ./Ac \
  --model-name regnety_400mf \
  --num-classes 2 \
  --epochs 100 \
  --batch-size 32 \
  --device cuda:0

关键参数说明：

参数名	含义	默认值
--data-path	数据集根目录路径（含 train/val/test 子文件夹）	D:\Regnet\学业预警\Ac
--model-name	RegNet 变体（regnety_400mf）	regnety_400mf
--num-classes	分类类别数	2
--epochs	训练轮数	100
--batch-size	批次大小（根据 GPU 显存调整）	32
--device	训练设备（cuda:0 或 cpu）	cuda:0

训练输出：

模型会自动保存验证集准确率最高的权重至 ./weights3_fast/ 目录，文件名为 best_model.pth；

TensorBoard 日志：运行 tensorboard --logdir runs；

训练日志（损失值、准确率）保存至 training_fast.txt。
```
### 6.2 模型预测
使用训练好的权重进行单张 GAF 图像预测，运行如下 Python 代码：

```bash
from PIL import Image
from torchvision import transforms
import torch
from model3 import create_regnet_with_attention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = create_regnet_with_attention("regnety_400mf", num_classes=2)
model.load_state_dict(torch.load("./weights3_fast/best_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

img = Image.open("sample.jpg").convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(input_tensor)
    prob = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(prob, dim=1).item()
    pred_class = "预警" if pred_idx == 1 else "达标"
    print(f"预测类别：{pred_class}，置信度：{prob[0, pred_idx]:.4f}")
```
预测输出示例：
```bash
输入图像路径：./sample.jpg
预测类别：达标
置信度：0.9972
```
### 6.3 数据集
可通过百度网盘获取完整文件：

通过网盘分享的文件：Ac.zip
链接: https://pan.baidu.com/s/1OwO38HIXWAkgRsrXJVGGew 
```bash
|-- Ac/
|   |-- train/
|   |   |-- E/
|   |   `-- NE/
|   |-- val/
|   |   |-- E/
|   |   `-- NE/
|   `-- test/
|       |-- E/
|       `-- NE/
```
适用场景：仅针对英语专业学生的“达标/预警”二分类。

## 9. 引用与联系方式
### 9.1 引用方式
论文已发表于 Information 期刊，请使用以下 BibTeX 格式引用：
```bash
bibtex
@article{ying2026english,
  title={English Proficiency Assessment Based on Improved RegNet and Dual Attention Mechanism},
  author={Ying, Y. and Wang, X. and Liu, M. and Xue, H. and Xu, L.},
  journal={Information},
  volume={17},
  number={386},
  pages={1--21},
  year={2026},
  publisher={MDPI}
}
```
### 9.2 联系方式
若遇到代码运行问题或学术交流需求，请联系：
邮箱：wangxiaowei@huuc.edu.cn

GitHub Issue：直接在本仓库提交 Issue，会在 1-3 个工作日内回复。
