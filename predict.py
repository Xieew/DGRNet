import os
import json
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from model3 import create_regnet_with_attention

# 显示缩放宽度（可根据需要调整）
DISPLAY_WIDTH = 800

# 尝试导入 OpenCV
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------- 新增：与训练一致的带Dropout模型包装 ----------
class ModelWithDropout(torch.nn.Module):
    """训练时使用的包装类，预测时仍可使用（eval模式下dropout自动关闭）"""
    def __init__(self, base_model, dropout_prob=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = torch.nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(self.base_model(x))
# ---------------------------------------------------------

def resize_for_display(img, target_width):
    """将图像按比例缩放到指定宽度，返回缩放后的图像和缩放因子"""
    w, h = img.size
    scale = target_width / w
    new_size = (target_width, int(h * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS), scale


def predict_one_image(img_path, model, transform, class_indict, device):
    """预测单张图像，显示图像并在左上角标注结果（放大显示）"""
    try:
        # 打开原始图像
        img_original = Image.open(img_path).convert('RGB')

        # 预处理并预测
        img_tensor = transform(img_original).unsqueeze(0).to(device)
        model.eval()  # 确保dropout等层处于评估模式
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_label = torch.argmax(probabilities).item()
            confidence = probabilities[pred_label].item()

        class_name = class_indict[str(pred_label)]
        print(f"图像路径: {img_path}")
        print(f"预测类别: {class_name} (索引: {pred_label})")
        print(f"置信度: {confidence:.4f}")

        # 准备显示的文本
        text = f"{class_name}: {confidence:.2%}"

        # 缩放图像用于显示
        img_display, scale = resize_for_display(img_original, DISPLAY_WIDTH)

        # --- 显示图像（优先 OpenCV）---
        if CV2_AVAILABLE:
            # 转换为 OpenCV 格式（BGR）
            img_cv = cv2.cvtColor(np.array(img_display), cv2.COLOR_RGB2BGR)
            # 根据缩放比例调整文字大小
            font_scale = 0.7 * scale  # 基础大小乘以缩放因子
            thickness = max(2, int(scale * 2))
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # 绘制黑色背景
            cv2.rectangle(img_cv, (5, 5), (5 + text_w + 10, 5 + text_h + 10), (0, 0, 0), -1)
            cv2.putText(img_cv, text, (10, 5 + text_h + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), thickness)
            # 显示窗口（窗口大小自适应）
            cv2.imshow('Prediction Result (Press any key to close)', img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # 使用 PIL 绘制
            draw = ImageDraw.Draw(img_display)
            # 根据缩放比例设置字体大小
            font_size = max(20, int(20 * scale))
            try:
                # 如果类别包含中文，请替换为系统中文字体路径，例如 "simhei.ttf"
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            # 计算文本尺寸并绘制背景
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle((0, 0, text_width + 10, text_height + 10), fill="black")
            draw.text((5, 5), text, fill="white", font=font)
            # 显示图像
            img_display.show()
            input("按回车键结束程序...")

        return class_name, confidence

    except Exception as e:
        print(f"预测失败: {e}")
        return None, None


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据预处理（与验证/测试阶段一致）
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别索引
    json_path = r'D:\Regnet\学业预警\class_indices.json'
    assert os.path.exists(json_path), f"文件 {json_path} 不存在"
    with open(json_path, "r") as f:
        class_indict = json.load(f)
    num_classes = len(class_indict)

    # 创建基础模型（与训练时相同的 backbone）
    base_model = create_regnet_with_attention(model_name="regnety_400mf", num_classes=num_classes)
    # 包装为带Dropout的模型（与训练完全一致）
    model = ModelWithDropout(base_model, dropout_prob=0.3).to(device)

    # 加载训练好的权重（路径指向训练脚本保存的最佳模型）
    # 注意：训练脚本 train3.py 中权重保存在 ./weights3_fast/best_model.pth
    model_weight_path = r"D:\Regnet\学业预警\weights3_fast\best_model.pth"  # 修改为实际路径
    assert os.path.exists(model_weight_path), f"权重文件 {model_weight_path} 不存在"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    print("模型权重加载成功！")

    # 预测单张图像
    image_path = r"D:\Regnet\学业预警\Ac\test\E\E1 (310).jpg"
    predict_one_image(image_path, model, data_transform, class_indict, device)


if __name__ == '__main__':
    main()