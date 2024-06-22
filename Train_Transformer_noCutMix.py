"""
Train_Transformer -

Author:霍畅
Date:2024/6/19
"""
import os
import timm
from datetime import datetime
import time
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision

# 数据集路径
dataroot = "..\data\cifar-10-batches-py"
# 文件保存路径
current = datetime.now()
run_time = current.strftime("%Y_%m_%d_%H_%M")
tb_dir = os.path.join('./logs', run_time)
writer = SummaryWriter(log_dir=tb_dir)
model_dir = os.path.join('./models', f"Resnet_NoCutMix_{run_time}.pth")
# 是否保存模型
save = 1


def test(net, testloader, criterion):
    total_loss = 0.0
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def train_and_test(net, trainloader, testloader, criterion, optimizer, epochs=10):
    best_accuracy = 0.0
    if save:
        torch.save(net.state_dict(), model_dir)
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        total = 0
        batches = len(trainloader)
        correct = 0
        for i, data in enumerate(trainloader, 0):
            # 应用CutMix
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 计算梯度
            optimizer.zero_grad()
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            # 计算并保存训练数据
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            record_step = 500  # 打印间隔step数
            if i % record_step == record_step - 1:
                train_accuracy = 100 * correct / total
                print(
                    f'[{epoch + 1}/{epochs}, {i + 1}/{batches}] loss: {running_loss / total:.3f} accuracy:{train_accuracy:.2f}%')
                iter = epoch * batches + (i + 1)
                writer.add_scalar('Loss/train', running_loss / total, iter)
                writer.add_scalar('Accuracy/train', train_accuracy, iter)
                running_loss = 0.0
                correct = 0
                total = 0
        test_loss, test_accuracy = test(net, testloader, criterion)
        best_accuracy = max(best_accuracy, test_accuracy)
        print(f'Test Loss: {test_loss:.3f}   Accuracy of epoch {epoch + 1}: {test_accuracy:.2f}%,')
        writer.add_scalar('Loss/test', test_loss, epoch + 1)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch + 1)
        if best_accuracy == test_accuracy and save:
            print(best_accuracy, "% model saved.")
            torch.save(net.state_dict(), model_dir)
    writer.close()


if __name__ == '__main__':
    # 设置超参数
    learning_rate = 0.001
    batch_size = 16
    total_epoch = 50
    weight_decay = 0.001
    num_classes = 10
    pretrained = False

    # 自定义DeiT模型参数
    embed_dim = 256  # 增加嵌入维度
    num_layers = 13  # 增加Transformer层数
    num_heads = 8  # 增加注意力头数量
    mlp_ratio = 4  # MLP比率保持不变
    img_size = 224  # 输入图像大小
    # 创建自定义DeiT模型
    model_cfg = {
        'img_size': img_size,
        'embed_dim': embed_dim,
        'depth': num_layers,
        'num_heads': num_heads,
        'mlp_ratio': mlp_ratio
    }
    # 加载DeiT-Tiny模型
    net = timm.create_model('deit_tiny_patch16_224', **model_cfg)
    # 修改最后的分类头以适应CIFAR-10数据集（10类）
    net.head = nn.Linear(net.head.in_features, num_classes)

    # # 加载resnet-18模型
    # net = models.resnet18(pretrained=pretrained)
    # net.fc = nn.Linear(net.fc.in_features, num_classes)

    # 确保使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    net = net.to(device)
    print("Network = ", net)

    # 数据预处理
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root=dataroot, train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root=dataroot, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    print("Data loaded successfully")
    print(f"Train set size {len(trainset)}")
    print(f"Test set size {len(testset)}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=30, gamma=0.1)

    # 训练网络
    print("Start Training")
    t1 = time.time()
    train_and_test(net, trainloader, testloader, criterion, optimizer, epochs=total_epoch)
    t2 = time.time()
    print('Finished Training')
    print(f"Total Time =   {t2 - t1:.3f} s")
    print(f"Average Time = {(t2 - t1) / total_epoch:.3f} s/epoch")
