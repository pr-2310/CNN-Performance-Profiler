!pip install torch torchvision gputil psutil humanize matplotlib

import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
import sys
import GPUtil
import os
import psutil
import humanize
import torch.cuda.profiler as ncu
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.cuda.memory as memory

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        start_time_ns = time.monotonic_ns()
        if START_TRACE:
            print("pc3427 1")
            ncu.start()
        x = self.conv2(x)
        if START_TRACE:
            print("pc3427 2")
            ncu.stop()
            quit()
        conv2_time_ns = time.monotonic_ns() - start_time_ns
        conv2_time_s = conv2_time_ns / 1e9
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output, conv2_time_s


def mem_report():
    memory_usage = {}
    print("CPU RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available))
    GPUs = GPUtil.getGPUs()
    for i, gpu in enumerate(GPUs):
        print(f'GPU {i}: Memory Free: {gpu.memoryFree}MB / {gpu.memoryTotal}MB | Utilization: {gpu.memoryUtil*100}%')
        memory_usage[f'GPU_{i}'] = (gpu.memoryTotal - gpu.memoryFree) / 1024  # Convert MB to GB
    return memory_usage

def main():
    # Settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 14
    lr = 1.0
    gamma = 0.7
    seed = 1

    use_cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    global START_TRACE
    START_TRACE=False
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # Model summary
    summary(model, (1, 28, 28))
    # Memory before training
    print('Memory Usage before training:')
    print(f"Allocated: {torch.cuda.memory_allocated(device=device)/1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device=device)/1024**3:.2f} GB")
    mem_report()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        START_TRACE=True
        test(model, device, test_loader)
        scheduler.step()



    print('Memory Usage after training:')
    print(f"Allocated: {torch.cuda.memory_allocated(device=device)/1024**3:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device=device)/1024**3:.2f} GB")
    mem_report()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_conv2_time_s = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, conv2_time_s = model(data)
        total_conv2_time_s += conv2_time_s  #
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print(f"Average time per batch for conv2: {total_conv2_time_s / len(train_loader)} seconds")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    main()