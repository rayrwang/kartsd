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
        self.fc1 = nn.Linear(8192, 500)
        self.fc2 = nn.Linear(500, 12120)

        # Networks
        self.n1 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 16, kernel_size=7, stride=(3, 4)),
            conv2=nn.Conv2d(16, 32, kernel_size=3, stride=2),
            conv3=nn.Conv2d(32, 64, kernel_size=3, stride=2),
            conv4=nn.Conv2d(64, 128, kernel_size=3, stride=2),
            act=nn.ReLU()
        ))
        self.n2 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 32, kernel_size=7, stride=(3, 6)),
            conv2=nn.Conv2d(32, 64, kernel_size=3, stride=2),
            conv3=nn.Conv2d(64, 128, kernel_size=3, stride=2),
            act=nn.ReLU()
        ))
        self.n3 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 32, kernel_size=7, stride=(3, 6)),
            conv2=nn.Conv2d(32, 64, kernel_size=3, stride=2),
            conv3=nn.Conv2d(64, 128, kernel_size=3, stride=2),
            act=nn.ReLU()
        ))

    def forward(self, x1, x2, x3):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3

        x1 = n1.act(n1.conv1(x1))
        x1 = n1.act(n1.conv2(x1))
        x1 = n1.act(n1.conv3(x1))
        x1 = n1.act(n1.conv4(x1))
        # print(x1.shape)
        x1 = x1.reshape(-1, 3584)

        x2 = n2.act(n2.conv1(x2))
        x2 = n2.act(n2.conv2(x2))
        x2 = n2.act(n2.conv3(x2))
        # print(x2.shape)
        x2 = x2.reshape(-1, 2304)

        x3 = n3.act(n3.conv1(x3))
        x3 = n3.act(n3.conv2(x3))
        x3 = n3.act(n3.conv3(x3))
        # print(x3.shape)
        x3 = x3.reshape(-1, 2304)

        x = torch.cat((x1, x2, x3), dim=1)

        # TODO Use transformer for decoding


        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
