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
        self.bn1 = nn.BatchNorm1d(5504)
        self.fc1 = nn.Linear(5504, 5504)
        self.fc2 = nn.Linear(5504, 12120)

        # Backbone and FPNs
        self.n1 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 32, kernel_size=(7, 9), stride=(1, 2), padding=(4, 4)),
            bn1=nn.BatchNorm2d(32),
            conv2=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            conv3=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            conv4=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            conv5=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            lateral=nn.Conv2d(32, 16, 1),
            bn2=nn.BatchNorm2d(16),
            up=nn.Upsample(scale_factor=2),
            outconv1=nn.Conv2d(16, 16, kernel_size=3),
            outconv2=nn.Conv2d(16, 16, kernel_size=3, stride=2),
            outconv3=nn.Conv2d(16, 16, kernel_size=3, stride=3),
            act=nn.ReLU()
        ))
        self.n2 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 32, kernel_size=(4, 3), stride=(1, 2), padding=(4, 9)),
            bn1=nn.BatchNorm2d(32),
            conv2=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            conv3=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            conv4=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            conv5=nn.Conv2d(32, 32, kernel_size=2, stride=2),
            lateral=nn.Conv2d(32, 16, 1),
            bn2=nn.BatchNorm2d(16),
            up=nn.Upsample(scale_factor=2),
            outconv1=nn.Conv2d(16, 16, kernel_size=3),
            outconv2=nn.Conv2d(16, 16, kernel_size=3, stride=2),
            outconv3=nn.Conv2d(16, 16, kernel_size=3, stride=3),
            act=nn.ReLU()
        ))
        self.n3 = self.n2

    def forward(self, x1, x2, x3):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3

        x1 = n1.act(n1.bn1(n1.conv1(x1)))
        # print(x1.shape)
        x1 = n1.act(n1.bn1(n1.conv2(x1)))
        c2 = x1.clone()
        # print(x1.shape)
        x1 = n1.act(n1.bn1(n1.conv3(x1)))
        c3 = x1.clone()
        # print(x1.shape)
        x1 = n1.act(n1.bn1(n1.conv4(x1)))
        c4 = x1.clone()
        # print(x1.shape)
        x1 = n1.act(n1.bn1(n1.conv5(x1)))
        c5 = x1.clone()
        # print(x1.shape)
        m5 = n1.act(n1.bn2(n1.lateral(c5)))
        p5 = n1.act(n1.bn2(n1.outconv1(m5)))
        m4 = n1.bn2(n1.up(m5)) + n1.act(n1.bn2(n1.lateral(c4)))
        p4 = n1.act(n1.bn2(n1.outconv2(m4)))
        m3 = n1.bn2(n1.up(m4)) + n1.act(n1.bn2(n1.lateral(c3)))
        p3 = n1.act(n1.bn2(n1.outconv3(m3)))
        # print(p5.shape, p4.shape, p3.shape)
        p5 = p5.reshape(-1, 672)
        p4 = p4.reshape(-1, 896)
        p3 = p3.reshape(-1, 1920)
        x1 = torch.cat((p5, p4, p3), dim=1)

        x2 = n2.act(n2.bn1(n2.conv1(x2)))
        # print(x2.shape)
        x2 = n2.act(n2.bn1(n2.conv2(x2)))
        c2 = x2.clone()
        # print(x2.shape)
        x2 = n2.act(n2.bn1(n2.conv3(x2)))
        c3 = x2.clone()
        # print(x2.shape)
        x2 = n2.act(n2.bn1(n2.conv4(x2)))
        c4 = x2.clone()
        # print(x2.shape)
        x2 = n2.act(n2.bn1(n2.conv5(x2)))
        c5 = x2.clone()
        # print(x2.shape)
        m5 = n2.act(n2.bn2(n2.lateral(c5)))
        p5 = n2.act(n2.bn2(n2.outconv1(m5)))
        m4 = n2.bn2(n2.up(m5)) + n2.act(n2.bn2(n2.lateral(c4)))
        p4 = n2.act(n2.bn2(n2.outconv2(m4)))
        m3 = n2.bn2(n2.up(m4)) + n2.act(n2.bn2(n2.lateral(c3)))
        p3 = n2.act(n2.bn2(n2.outconv3(m3)))
        # print(p5.shape, p4.shape, p3.shape)
        p5 = p5.reshape(-1, 128)
        p4 = p4.reshape(-1, 240)
        p3 = p3.reshape(-1, 640)
        x2 = torch.cat((p5, p4, p3), dim=1)

        x3 = n3.act(n3.bn1(n3.conv1(x3)))
        # print(x2.shape)
        x3 = n3.act(n3.bn1(n3.conv2(x3)))
        c2 = x3.clone()
        # print(x2.shape)
        x3 = n3.act(n3.bn1(n3.conv3(x3)))
        c3 = x3.clone()
        # print(x2.shape)
        x3 = n3.act(n3.bn1(n3.conv4(x3)))
        c4 = x3.clone()
        # print(x2.shape)
        x3 = n3.act(n3.bn1(n3.conv5(x3)))
        c5 = x3.clone()
        # print(x2.shape)
        m5 = n3.act(n3.bn2(n3.lateral(c5)))
        p5 = n3.act(n3.bn2(n3.outconv1(m5)))
        m4 = n3.bn2(n3.up(m5)) + n3.act(n3.bn2(n3.lateral(c4)))
        p4 = n3.act(n3.bn2(n3.outconv2(m4)))
        m3 = n3.bn2(n3.up(m4)) + n3.act(n3.bn2(n3.lateral(c3)))
        p3 = n3.act(n3.bn2(n3.outconv3(m3)))
        # print(p5.shape, p4.shape, p3.shape)
        p5 = p5.reshape(-1, 128)
        p4 = p4.reshape(-1, 240)
        p3 = p3.reshape(-1, 640)
        x3 = torch.cat((p5, p4, p3), dim=1)

        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.fc1(x))
        x = self.bn1(self.dropout(x))
        x = F.relu(self.fc2(x))

        return x
