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

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(3, 10, kernel_size=11, stride=4)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(2160, 2160)
        self.fc2 = nn.Linear(2160, 12120)

        # # ResNets
        # self.r1 = nn.ModuleDict(dict(
        #     pool=nn.MaxPool2d(kernel_size=3, stride=2),
        #     conv1=nn.Conv2d(3, 50, kernel_size=7, stride=2, padding=0),
        #     bn1=nn.BatchNorm2d(50),
        #     conv2=nn.Conv2d(50, 50, kernel_size=3, padding="same"),
        #     conv3=nn.Conv2d(50, 100, kernel_size=3, padding="same"),
        #     skip1=nn.Conv2d(50, 100, kernel_size=1, padding="same"),
        #     conv4=nn.Conv2d(100, 100, kernel_size=3, padding="same"),
        #     bn2=nn.BatchNorm2d(100),
        #     act=nn.ReLU()
        # ))
        #
        # self.r2 = self.r1
        # self.r3 = self.r1

    def forward(self, x1, x2, x3):
        # r1 = self.r1
        # x1 = r1.pool(r1.act(r1.conv1(x1)))
        # # x1 = r1.bn1(x1)
        # skip = x1
        # x1 = r1.pool(r1.act(r1.conv2(r1.act(r1.conv2(x1))) + skip))
        # # x1 = r1.bn1(x1)
        # skip = r1.skip1(x1)
        # x1 = r1.pool(r1.act(r1.conv4(r1.act(r1.conv3(x1))) + skip))
        # # x1 = r1.bn2(x1)
        # skip = x1
        # x1 = r1.pool(r1.act(r1.conv4(r1.act(r1.conv4(x1))) + skip))
        # # x1 = r1.bn2(x1)
        # # print(x1.shape)
        # x1 = x1.reshape(-1, 1800)
        #
        # r2 = self.r2
        # x2 = r2.pool(r2.act(r2.conv1(x2)))
        # # x2 = r2.bn1(x2)
        # skip = x2
        # x2 = r2.act(r2.conv2(r2.act(r2.conv2(x2))) + skip)
        # # x2 = r2.bn1(x2)
        # skip = r2.skip1(x2)
        # x2 = r2.pool(r2.act(r2.conv4(r2.act(r2.conv3(x2))) + skip))
        # # x2 = r2.bn2(x2)
        # skip = x2
        # x2 = r2.pool(r2.act(r2.conv4(r2.act(r2.conv4(x2))) + skip))
        # # x2 = r2.bn2(x2)
        # # print(x2.shape)
        # x2 = x2.reshape(-1, 1800)
        #
        # r3 = self.r3
        # x3 = r3.pool(r3.act(r3.conv1(x3)))
        # # x3 = r1.bn1(x3)
        # skip = x3
        # x3 = r3.act(r3.conv2(r3.act(r3.conv2(x3))) + skip)
        # # x3 = r1.bn1(x3)
        # skip = r3.skip1(x3)
        # x3 = r3.pool(r3.act(r3.conv4(r3.act(r3.conv3(x3))) + skip))
        # # x3 = r1.bn2(x3)
        # skip = x3
        # x3 = r3.pool(r3.act(r3.conv4(r3.act(r3.conv4(x3))) + skip))
        # # x3 = r3.bn2(x3)
        # # print(x2.shape)
        # x3 = x3.reshape(-1, 1800)

        x1 = self.pool(F.relu(self.conv1(x1)))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = F.relu(self.conv3(x1))
        x1 = self.pool(F.relu(self.conv4(x1)))
        # print(x1.shape)
        x1 = x1.reshape(-1, 720)
        # print(x1.shape)

        x2 = self.pool(F.relu(self.conv1(x2)))
        x2 = F.relu(self.conv2(x2))
        x2 = F.relu(self.conv3(x2))
        x2 = self.pool(F.relu(self.conv4(x2)))
        # print(x2.shape)
        x2 = x2.reshape(-1, 720)
        # print(x2.shape)

        x3 = self.pool(F.relu(self.conv1(x3)))
        x3 = F.relu(self.conv2(x3))
        x3 = F.relu(self.conv3(x3))
        x3 = self.pool(F.relu(self.conv4(x3)))
        # print(x3.shape)
        x3 = x3.reshape(-1, 720)
        # print(x3.shape)

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
