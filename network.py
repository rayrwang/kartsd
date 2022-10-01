import torch
import torch.nn as nn
import torch.nn.functional as f


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
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.reshape(-1, 350)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class VSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 10, 5, stride=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5, stride=2)
        self.fc1 = nn.Linear(1220, 1220)
        self.fc2 = nn.Linear(1220, 12120)

    def forward(self, x1, x2, x3):
        x1 = self.pool(f.relu(self.conv1(x1)))
        x1 = self.pool(f.relu(self.conv2(x1)))
        # print(x1.shape)
        x1 = x1.reshape(-1, 420)
        # print(x1.shape)

        x2 = self.pool(f.relu(self.conv1(x2)))
        x2 = self.pool(f.relu(self.conv2(x2)))
        # print(x2.shape)
        x2 = x2.reshape(-1, 400)
        # print(x2.shape)

        x3 = self.pool(f.relu(self.conv1(x3)))
        x3 = self.pool(f.relu(self.conv2(x3)))
        # print(x3.shape)
        x3 = x3.reshape(-1, 400)
        # print(x3.shape)

        x = torch.cat((x1, x2, x3), dim=1)
        x = f.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

