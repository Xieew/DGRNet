import os
import argparse
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from model3 import create_regnet_with_attention
from my_dataset import MyDataSet
from utils import evaluate   # 假设 evaluate 函数已存在

# ---------- 工具函数 ----------
def get_images_and_labels(data_path):
    classes = os.listdir(data_path)
    images_path, images_label = [], []
    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_path, cls)
        images = [os.path.join(cls_path, img) for img in os.listdir(cls_path)]
        images_path.extend(images)
        images_label.extend([i] * len(images))
    return images_path, images_label

def calculate_metrics(model, data_loader, device, criterion):
    """普通前向计算损失和准确率"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / total, correct / total

def validate_with_tta(model, loader, device):
    """测试时增强（原图+水平翻转），提高评估稳定性"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs_orig = model(images)
            images_flip = torch.flip(images, dims=[3])
            outputs_flip = model(images_flip)
            avg_outputs = (outputs_orig + outputs_flip) / 2
            _, predicted = torch.max(avg_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def print_class_distribution(loader, name):
    labels = [lbl for _, lbl in loader.dataset]
    unique, counts = torch.unique(torch.tensor(labels), return_counts=True)
    print(f"\n{name} 类别分布:")
    for u, c in zip(unique, counts):
        print(f"  类别 {u.item()}: {c.item()} 样本")

# ---------- 带Dropout的模型包装 ----------
class ModelWithDropout(nn.Module):
    def __init__(self, base_model, dropout_prob=0.3):
        super().__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        return self.dropout(self.base_model(x))

# ---------- 训练函数（单次前向，无TTA）----------
def train_one_epoch_fast(model, optimizer, data_loader, device, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            # 梯度裁剪（可选）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples

# ---------- 主函数 ----------
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 清理旧权重（可选）
    weights_dir = "./weights3_fast"
    os.makedirs(weights_dir, exist_ok=True)

    print(args)
    tb_writer = SummaryWriter()
    log_file = open("./training_fast.txt", "w", encoding="utf-8")
    log_file.write("epoch\ttrain_loss\tval_loss\tval_acc\ttest_loss\ttest_acc\tlr\n")

    # 数据集
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')
    test_path = os.path.join(args.data_path, 'test')
    train_images, train_labels = get_images_and_labels(train_path)
    val_images, val_labels = get_images_and_labels(val_path)
    test_images, test_labels = get_images_and_labels(test_path)

    # 数据增强（训练集增强适度，验证/测试只做基本预处理）
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = MyDataSet(train_images, train_labels, data_transform["train"])
    val_dataset = MyDataSet(val_images, val_labels, data_transform["val"])
    test_dataset = MyDataSet(test_images, test_labels, data_transform["val"])  # 测试集同验证

    nw = min(os.cpu_count(), args.batch_size, 8)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, pin_memory=True,
                                               num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True,
                                              num_workers=nw, collate_fn=test_dataset.collate_fn)

    print_class_distribution(train_loader, "训练集")
    print_class_distribution(val_loader, "验证集")
    print_class_distribution(test_loader, "测试集")

    # 模型
    base_model = create_regnet_with_attention(model_name=args.model_name, num_classes=args.num_classes)
    if args.weights:
        weights_dict = torch.load(args.weights, map_location='cpu')
        # 简单处理权重键名（根据你的实际模型结构调整）
        model_dict = base_model.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        base_model.load_state_dict(pretrained_dict, strict=False)
        print("预训练权重加载完成")

    model = ModelWithDropout(base_model, dropout_prob=0.3).to(device)

    # 优化器与学习率调度（降低初始lr以配合快速训练）
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 混合精度缩放器
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_val_acc = 0.0
    best_val_loss = float('inf')
    eval_interval = 1  # 每5个epoch计算一次训练集准确率

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_one_epoch_fast(model, optimizer, train_loader, device, criterion, scaler)

        # 每eval_interval个epoch或在最后epoch计算训练集准确率
        if epoch % eval_interval == 0 or epoch == args.epochs - 1:
            train_acc = evaluate(model, train_loader, device)
        else:
            train_acc = None

        # 验证（始终使用TTA）
        val_loss, _ = calculate_metrics(model, val_loader, device, criterion)
        val_acc = validate_with_tta(model, val_loader, device)

        # 测试（普通前向，可改为TTA以保持一致，但为了速度，仍用普通）
        test_loss, test_acc = calculate_metrics(model, test_loader, device, criterion)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # 打印日志
        log_msg = (f"[epoch {epoch}] train_loss: {train_loss:.4f}, "
                   f"{'train_acc: {:.4f}, '.format(train_acc) if train_acc is not None else ''}"
                   f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}, "
                   f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, lr: {current_lr:.6f}")
        print(log_msg)

        # TensorBoard
        tb_writer.add_scalar("train_loss", train_loss, epoch)
        if train_acc is not None:
            tb_writer.add_scalar("train_accuracy", train_acc, epoch)
        tb_writer.add_scalar("val_loss", val_loss, epoch)
        tb_writer.add_scalar("val_accuracy", val_acc, epoch)
        tb_writer.add_scalar("test_loss", test_loss, epoch)
        tb_writer.add_scalar("test_accuracy", test_acc, epoch)
        tb_writer.add_scalar("learning_rate", current_lr, epoch)

        # 保存最佳模型（按验证准确率）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(weights_dir, "best_model.pth"))
            print(f"保存最佳模型，验证准确率: {best_val_acc:.4f}")

        # 日志写入文件
        log_file.write(f"{epoch}\t{train_loss:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{test_loss:.4f}\t{test_acc:.4f}\t{current_lr:.6f}\n")
        log_file.flush()

    # 最终测试
    print("\n训练完成，加载最佳模型评估测试集...")
    model.load_state_dict(torch.load(os.path.join(weights_dir, "best_model.pth")))
    test_loss, test_acc = calculate_metrics(model, test_loader, device, criterion)
    print(f"最佳模型测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")

    log_file.close()
    tb_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-path', type=str, default=r"D:\Regnet\学业预警\Ac")
    parser.add_argument('--model-name', default='regnety_400mf')
    parser.add_argument('--weights', type=str, default=r'D:\Regnet\学业预警\regnety_400mf.pth')
    parser.add_argument('--device', default='cuda:0')
    opt = parser.parse_args()
    main(opt)