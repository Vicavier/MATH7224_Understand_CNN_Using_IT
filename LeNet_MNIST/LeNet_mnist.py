from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 输入通道1，输出通道6，卷积核大小5
        self.conv2 = nn.Conv2d(6, 12, 5)
        self.fc1 = nn.Linear(12 * 4 * 4, 120)  # 卷积层输出特征图大小为4x4
        self.fc2 = nn.Linear(120, 10)  # 输出层，10个类别

    def forward(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)  # 池化层，大小2
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(-1, 12 * 4 * 4)  # 展平特征图
        x = self.fc2(self.fc1(x))
        return x

    def layer_outputs(self, x):
        x1 = self.conv1(x)
        x2 = torch.relu(x1)
        x3 = torch.max_pool2d(x1, 2)
        x4 = self.conv2(x3)
        x5 = torch.relu(x4)
        x6 = torch.max_pool2d(x5, 2)
        return x1, x2, x3, x4, x5, x6

    def classifier_outputs(self, x):
        x = torch.max_pool2d(torch.relu(self.conv1(x)), 2)
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)
        x = x.view(-1, 12 * 4 * 4)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = torch.softmax(x2, dim=1)
        return x1, x2, x3


# 训练模型
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 测试模型
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # 实例化模型
    model = LeNet()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

    # 训练和测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_dir = Path("./model_checkpoints/lenet_mnist")
    model_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 11):  # 训练10个epoch
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        torch.save(model.state_dict(), model_dir / f"lenet_mnist_ep{epoch}.pth")
