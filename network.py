import torch
import torch.nn as nn
import torch.nn.functional as F


class SteerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 5, 5, stride=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 10, 5, stride=2)
        self.fc1 = nn.Linear(350, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 350)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class VSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3440, 3440)
        self.fc2 = nn.Linear(3440, 12120)

        # ResNets
        self.r1 = nn.ModuleDict(dict(
            pool=nn.MaxPool2d(kernel_size=2, stride=2),
            conv1=nn.Conv2d(3, 10, kernel_size=7, stride=(1, 2)),
            bn1=nn.BatchNorm2d(10),
            conv2=nn.Conv2d(10, 20, kernel_size=5, stride=1),
            bn2=nn.BatchNorm2d(20),
            conv3=nn.Conv2d(20, 20, kernel_size=3, stride=1),
            conv4=nn.Conv2d(20, 40, kernel_size=3, stride=1),
            bn3=nn.BatchNorm2d(40),
            act=nn.ReLU()
        ))
        self.r2 = nn.ModuleDict(dict(
            pool=nn.MaxPool2d(kernel_size=2, stride=2),
            conv1=nn.Conv2d(3, 10, kernel_size=(5, 7), stride=(1, 2)),
            bn1=nn.BatchNorm2d(10),
            conv2=nn.Conv2d(10, 20, kernel_size=(3, 5), stride=1),
            bn2=nn.BatchNorm2d(20),
            conv3=nn.Conv2d(20, 20, kernel_size=3, stride=1),
            conv4=nn.Conv2d(20, 40, kernel_size=3, stride=1),
            bn3=nn.BatchNorm2d(40),
            act=nn.ReLU()
        ))
        self.r3 = self.r2

    def forward(self, x1, x2, x3):
        r1 = self.r1
        r2 = self.r2
        r3 = self.r3

        x1 = F.relu(r1.bn1(r1.pool(r1.conv1(x1))))
        x1 = F.relu(r1.bn2(r1.pool(r1.conv2(x1))))
        x1 = F.relu(r1.bn2(r1.pool(r1.conv3(x1))))
        x1 = F.relu(r1.bn3(r1.pool(r1.conv4(x1))))
        # print(x1.shape)
        x1 = x1.reshape(-1, 1200)
        # print(x1.shape)

        x2 = F.relu(r2.bn1(r2.pool(r2.conv1(x2))))
        x2 = F.relu(r2.bn2(r2.pool(r2.conv2(x2))))
        x2 = F.relu(r2.bn2(r2.conv3(x2)))
        x2 = F.relu(r2.bn3(r2.pool(r2.conv4(x2))))
        # print(x2.shape)
        x2 = x2.reshape(-1, 1120)
        # print(x2.shape)

        x3 = F.relu(r3.bn1(r3.pool(r3.conv1(x3))))
        x3 = F.relu(r3.bn2(r3.pool(r3.conv2(x3))))
        x3 = F.relu(r3.bn2(r3.conv3(x3)))
        x3 = F.relu(r3.bn3(r3.pool(r3.conv4(x3))))
        # print(x3.shape)
        x3 = x3.reshape(-1, 1120)
        # print(x3.shape)

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
