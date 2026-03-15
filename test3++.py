import os
import json
import torch
from PIL import Image
from torchvision import transforms
from model3 import create_regnet_with_attention
import datetime


def get_images_and_labels(data_path):
    """读取文件夹中的图像路径和标签，确保按固定顺序读取四类数据"""
    # 显式指定四类别的文件夹名称（根据实际情况修改为你的四类名称）
    # 确保与训练时的类别顺序一致
    classes = ["E", "NE"]  # 替换为你的四类名称
    images_path = []
    images_label = []

    # 验证文件夹是否存在
    for cls in classes:
        cls_path = os.path.join(data_path, cls)
        if not os.path.exists(cls_path):
            raise ValueError(f"类别文件夹不存在: {cls_path}，请检查路径是否正确")

    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        # 只读取图像文件
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)
                  if img.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        if not images:
            print(f"警告：类别 {cls} 文件夹中未找到图像文件")
        images_path.extend(images)
        images_label.extend([i] * len(images))

    # 验证是否有四类数据
    unique_labels = set(images_label)
    if len(unique_labels) != 2:
        print(f"警告：实际加载的类别数为 {len(unique_labels)}，预期为4类")

    return images_path, images_label


def batch_predict(model, data_transform, test_images_path, test_images_label, class_indict, device):
    """批量预测函数，返回详细结果和统计信息"""
    num_classes = 2
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    detailed_results = []

    # 预测整个测试集
    for img_idx, (img_path, true_label) in enumerate(zip(test_images_path, test_images_label)):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = data_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(img_tensor)
                # 使用softmax获取概率
                probabilities = torch.softmax(output, dim=1)
                _, pred = torch.max(output, 1)
                pred_label = pred.item()

                # 获取所有类别的概率
                prob_values = probabilities[0].cpu().numpy()

            # 确保标签在0-3范围内
            assert 0 <= true_label < 2, f"无效的真实标签 {true_label}，必须在0-3之间"
            class_total[true_label] += 1

            # 检查预测是否正确
            is_correct = pred_label == true_label
            if is_correct:
                class_correct[true_label] += 1

            # 获取类别名称
            true_class_name = class_indict.get(str(true_label), f"未知类别{true_label}")
            pred_class_name = class_indict.get(str(pred_label), f"未知类别{pred_label}")

            # 存储详细结果
            result_info = {
                'index': img_idx + 1,
                'image_path': img_path,
                'true_label': true_label,
                'true_class': true_class_name,
                'pred_label': pred_label,
                'pred_class': pred_class_name,
                'is_correct': is_correct,
                'probabilities': prob_values,
                'max_probability': prob_values[pred_label]  # 预测类别的概率
            }
            detailed_results.append(result_info)

            # 打印每张图片的预测结果
            print(f"图片 {img_idx + 1:3d}/{len(test_images_path)}: {os.path.basename(img_path)}")
            print(f"  真实类别: {true_class_name} (标签: {true_label})")
            print(f"  预测类别: {pred_class_name} (标签: {pred_label})")
            print(f"  是否正确: {'是' if is_correct else '否'}")
            print(f"  预测概率: {prob_values[pred_label]:.4f}")
            print(f"  各类别概率:")
            for i in range(num_classes):
                class_name = class_indict.get(str(i), f"未知类别{i}")
                print(f"    {class_name}: {prob_values[i]:.4f}")
            print("-" * 50)

        except Exception as e:
            print(f"处理 {img_path} 时出错: {e}")

    return detailed_results, class_correct, class_total


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "GPU" if torch.cuda.is_available() else "CPU"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Starting prediction on {device_type} at {current_time}")

    # 数据预处理（与训练时验证集保持一致）
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载类别索引（确保包含4类）
    json_path = r'D:\Regnet\学业预警\class_indices.json'  # 确保该文件包含4类
    assert os.path.exists(json_path), f"类别索引文件 {json_path} 不存在"
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 验证类别数是否为4
    assert len(class_indict) == 2, f"类别索引文件中包含 {len(class_indict)} 类，预期为4类"
    num_classes = 2  # 显式指定为4类

    # 创建模型（指定4类）
    model = create_regnet_with_attention(
        model_name="regnety_400mf",
        num_classes=num_classes  # 确保输出为4类
    ).to(device)
    # 模型权重路径
    model_weight_path = r"D:\Regnet\学业预警\weights3_fast\best_model.pth"
    assert os.path.exists(model_weight_path), f"权重文件 {model_weight_path} 不存在"

    # 加载权重并处理键名前缀
    state_dict = torch.load(model_weight_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('base_model.'):
            new_key = k[len('base_model.'):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"模型加载自: {model_weight_path}")

    # 测试数据路径（确保包含4个子文件夹）
    test_data_path = r"D:\Regnet\学业预警\Ac\test"  # 与train3.py中的测试集路径保持一致
    assert os.path.exists(test_data_path), f"测试数据路径 {test_data_path} 不存在"
    test_images_path, test_images_label = get_images_and_labels(test_data_path)
    print(f"在 {test_data_path} 中找到 {len(test_images_path)} 张图像")

    # 创建结果目录
    results_dir = r"D:\Regnet\学业预警\regnet_xueyeyujing\results-best(3++)"
    os.makedirs(results_dir, exist_ok=True)

    # 更新输出文件变量名
    original_output_file = os.path.join(results_dir, "4.txt")  # 原始详细预测结果
    probability_output_file = os.path.join(results_dir, "3.txt")  # 概率预测结果

    # 批量预测
    detailed_results, class_correct, class_total = batch_predict(
        model, data_transform, test_images_path, test_images_label, class_indict, device
    )

    # 将原始详细结果写入4.txt - 只包含指定的列
    with open(original_output_file, 'w', encoding='utf-8') as f:
        f.write("原始详细预测结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"日期: {current_time}\n")
        f.write(f"设备: {device_type}\n")
        f.write(f"使用模型权重: {os.path.basename(model_weight_path)}\n")
        f.write(f"测试数据集: {test_data_path}\n")
        f.write(f"总图像数: {len(test_images_path)}\n\n")

        # 写入表头 - 只包含指定的列
        f.write(f"{'真实类别':<12} {'预测类别':<12} {'预测概率':<12} {'是否正确':<8}\n")
        f.write("-" * 60 + "\n")

        # 按顺序写入每张图片的详细结果 - 只包含指定的列
        for result in detailed_results:
            f.write(f"{result['true_class']:<12} {result['pred_class']:<12} ")
            f.write(f"{result['max_probability']:.6f}    ")
            f.write(f"{'是' if result['is_correct'] else '否':<8}\n")

    print(f"原始详细预测结果已保存到: {original_output_file}")

    # 将概率预测结果写入3.txt - 只包含指定的列
    with open(probability_output_file, 'w', encoding='utf-8') as f:
        f.write("概率预测结果\n")
        f.write("=" * 60 + "\n")
        f.write(f"日期: {current_time}\n")
        f.write(f"设备: {device_type}\n")
        f.write(f"使用模型权重: {os.path.basename(model_weight_path)}\n")
        f.write(f"测试数据集: {test_data_path}\n")
        f.write(f"总图像数: {len(test_images_path)}\n\n")

        # 写入表头 - 只包含指定的列
        f.write(f"{'真实类别':<12} {'预测类别':<12} {'预测概率':<12} {'是否正确':<8}\n")
        f.write("-" * 60 + "\n")

        # 按顺序写入每张图片的概率结果 - 只包含指定的列
        for result in detailed_results:
            f.write(f"{result['true_class']:<12} {result['pred_class']:<12} ")
            f.write(f"{result['max_probability']:.6f}    ")
            f.write(f"{'是' if result['is_correct'] else '否':<8}\n")

    print(f"概率预测结果已保存到: {probability_output_file}")

    # 计算并输出准确率
    avg_accuracy = 0.0
    accuracy_lines = []

    print("\n类别准确率:")
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
        else:
            accuracy = 0.0
        class_name = class_indict.get(str(i), f"未知类别{i}")
        stats_line = f"{class_name:15}: {accuracy:.2%} ({class_correct[i]}/{class_total[i]})"
        print(stats_line)
        accuracy_lines.append(stats_line)
        avg_accuracy += accuracy

    avg_accuracy /= num_classes  # 除以4类计算平均准确率
    print(f"\n平均准确率: {avg_accuracy:.2%}")

    # 在两个文件末尾添加准确率统计
    for output_file in [original_output_file, probability_output_file]:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("准确率统计:\n")
            f.write("=" * 60 + "\n")
            for line in accuracy_lines:
                f.write(line + "\n")
            f.write(f"\n平均准确率: {avg_accuracy:.2%}\n")
            f.write("=" * 60 + "\n")

    print(f"\n程序执行完成，生成了两个输出文件：")
    print(f"1. {original_output_file} - 保存原始详细预测结果")
    print(f"2. {probability_output_file} - 保存概率预测结果")
    print(f"两个文件都包含以下列：真实类别、预测类别、预测概率、是否正确")


if __name__ == '__main__':
    main()