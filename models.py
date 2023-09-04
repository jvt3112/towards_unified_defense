from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
"""
Code used from: https://github.com/ssg-research/conflicts-in-ml-protection-mechanisms/blob/main/src/models.py
- MNIST_CNN 
- MNIST_SMALL_CNN
- CIFAR_CNN

Code used for ResNet CIFAR10/100 implementation from: https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py , 
https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-mnist.ipynb
"""

class MNIST_CNN(nn.Module):
    def __init__(self, args):
        super(MNIST_CNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),

            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512),

            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10))

    def forward(self, x):
        out = self.net(x)
        return out

class MNIST_CNN_SMALL(nn.Module):
    def __init__(self,args) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2), #Output: (13, 13, 16)
            nn.MaxPool2d(2, stride=1), #Output (12, 12, 16)
            nn.Tanh(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0), #Output (5, 5, 32)
            nn.MaxPool2d(2, stride=1), #Output (4, 4, 32)
            nn.Tanh(),

            nn.Flatten(),

            nn.Linear(4 * 4 * 32, 32),
            nn.Tanh(),

            nn.Linear(32, 10))

    def forward(self, x):
        out = self.net(x)
        return out

class CIFAR_CNN(nn.Module):
    """
    An 8-layer CNN with Tanh activations from papernot et al. paper
    """
    def __init__(self,args) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 32, 32)
            nn.Tanh(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), # Output: (32, 32, 32)
            nn.MaxPool2d(2, stride=2), # Output: (16, 16, 32)
            nn.Tanh(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # Output: (16, 16, 64)
            nn.Tanh(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), # Output: (16, 16, 64)
            nn.MaxPool2d(2, stride=2), # Output: (8, 8, 64)
            nn.Tanh(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Output: (8, 8, 128)
            nn.Tanh(),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), # Output: (8, 8, 128)
            nn.MaxPool2d(2, stride=2), # Output: (4, 4, 128)
            nn.Tanh(),

            nn.Flatten(),

            nn.Linear(4 * 4 * 128, 128),
            nn.Tanh(),

            nn.Linear(128, 10))

    def forward(self, x):
        out = self.net(x)
        return out

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # out = F.avg_pool2d(out, out.size()[3])
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def resnet18(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet20(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [3, 3, 3], num_classes)


def resnet32(num_classes: int) -> nn.Module:
    return ResNet(BasicBlock, [5, 5, 5], num_classes)