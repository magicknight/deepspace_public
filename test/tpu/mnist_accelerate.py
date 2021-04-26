from accelerate import Accelerator
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import os
import numpy as np
import math
from matplotlib import pyplot as plt
import time

plt.rcParams['figure.figsize'] = [12, 8]

M, N = 4, 6
RESULT_IMG_PATH = './test_result.png'
# Define Parameters
FLAGS = {}
FLAGS['datadir'] = "/tmp/mnist"
FLAGS['batch_size'] = 128
FLAGS['num_workers'] = 4
FLAGS['learning_rate'] = 0.01
FLAGS['momentum'] = 0.5
FLAGS['num_epochs'] = 10
FLAGS['num_cores'] = 8
FLAGS['log_steps'] = 20
FLAGS['metrics_debug'] = False


def plot_results(images, labels, preds):
    images, labels, preds = images[:M*N], labels[:M*N], preds[:M*N]
    inv_norm = transforms.Normalize((-0.1307/0.3081,), (1/0.3081,))

    num_images = images.shape[0]
    fig, axes = plt.subplots(M, N, figsize=(11, 9))
    fig.suptitle('Correct / Predicted Labels (Red text for incorrect ones)')

    for i, ax in enumerate(fig.axes):
        ax.axis('off')
        if i >= num_images:
            continue
        img, label, prediction = images[i], labels[i], preds[i]
        img = inv_norm(img)
        img = img.squeeze()  # [1,Y,X] -> [Y,X]
        label, prediction = label.item(), prediction.item()
        if label == prediction:
            ax.set_title(u'\u2713', color='blue', fontsize=22)
        else:
            ax.set_title(
                'X {}/{}'.format(label, prediction), color='red')
        ax.imshow(img)
    plt.savefig(RESULT_IMG_PATH, transparent=True)


accelerator = Accelerator()
device = accelerator.device
print('device is ', device)


class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(20)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.bn1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Only instantiate model weights once in memory.
# WRAPPED_MODEL = MNIST()


def train_mnist():
    torch.manual_seed(1)

    def get_dataset():
        norm = transforms.Normalize((0.1307,), (0.3081,))
        train_dataset = datasets.MNIST(
            FLAGS['datadir'],
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), norm]))
        test_dataset = datasets.MNIST(
            FLAGS['datadir'],
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), norm]))

        return train_dataset, test_dataset

    # Using the serial executor avoids multiple processes to
    # download the same data.
    train_dataset, test_dataset = get_dataset()

    train_loader = DataLoader(
        train_dataset,
        batch_size=FLAGS['batch_size'],
        num_workers=FLAGS['num_workers'],
        drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=FLAGS['batch_size'],
        shuffle=False,
        num_workers=FLAGS['num_workers'],
        drop_last=True)

    # Scale learning rate to world size
    lr = FLAGS['learning_rate']
    model = MNIST().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=FLAGS['momentum'])
    loss_fn = nn.NLLLoss()
    model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

    def train_loop_fn(loader):
        model.train()
        for x, (data, target) in enumerate(loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            accelerator.backward(loss)
            optimizer.step()

    def test_loop_fn(loader):
        total_samples = 0
        correct = 0
        model.eval()
        data, pred, target = None, None, None
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size()[0]

        accuracy = 100.0 * correct / total_samples
        print('Accuracy={:.2f}%'.format(
            accuracy), flush=True)
        return accuracy, data, pred, target

    # Train and eval loops
    accuracy = 0.0
    data, pred, target = None, None, None
    for epoch in range(1, FLAGS['num_epochs'] + 1):
        train_loop_fn(train_loader)
        print("Finished training epoch {}".format(epoch))

        accuracy, data, pred, target = test_loop_fn(test_loader)
    return accuracy, data, pred, target


def main():
    start_time = time.time()
    accuracy, data, pred, target = train_mnist()
    print("--- %s seconds ---" % (time.time() - start_time))
    plot_results(data.cpu(), pred.cpu(), target.cpu())


if __name__ == '__main__':
    main()
