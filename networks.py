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
        self.fc1 = nn.Linear(9472, 1000)
        self.fc2 = nn.Linear(1000, 12120)

        # ResNet and FPNs
        self.n1 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 16, kernel_size=(3, 5), stride=(3, 4), padding=1, bias=False),
            pool=nn.MaxPool2d(kernel_size=2, stride=2),
            skip1=nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            conv2=nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            conv3=nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=False),
            skip2=nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            conv4=nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False),
            conv5=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same", bias=False),
            skip3=nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            conv6=nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
            conv7=nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same", bias=False),
            lateral1=nn.Conv2d(16, 16, 1, bias=False),
            lateral2=nn.Conv2d(32, 32, 1, bias=False),
            lateral3=nn.Conv2d(64, 64, 1, bias=False),
            lateral4=nn.Conv2d(128, 128, 1, bias=False),
            down=nn.Upsample(scale_factor=2),
            down1=nn.Conv2d(128, 64, 1, bias=False),
            down2=nn.Conv2d(64, 32, 1, bias=False),
            down3=nn.Conv2d(32, 16, 1, bias=False),
            outconv1=nn.Conv2d(128, 64, kernel_size=3, stride=3, bias=False),
            outconv2=nn.Conv2d(64, 32, kernel_size=3, stride=3, bias=False),
            outconv3=nn.Conv2d(32, 16, kernel_size=3, stride=3, bias=False),
            outconv4=nn.Conv2d(16, 8, kernel_size=3, stride=3, bias=False),
            act=nn.ReLU()
        ))
        self.n2 = nn.ModuleDict(dict(
            conv1=nn.Conv2d(3, 16, kernel_size=(3, 4), stride=(2, 4), padding=(3, 8), bias=False),
            pool=nn.MaxPool2d(kernel_size=2, stride=2),
            skip1=nn.Conv2d(16, 32, kernel_size=1, stride=1, bias=False),
            conv2=nn.Conv2d(16, 32, kernel_size=2, stride=2, bias=False),
            conv3=nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same", bias=False),
            skip2=nn.Conv2d(32, 64, kernel_size=1, stride=1, bias=False),
            conv4=nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False),
            conv5=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding="same", bias=False),
            skip3=nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            conv6=nn.Conv2d(64, 128, kernel_size=2, stride=2, bias=False),
            conv7=nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same", bias=False),
            lateral1=nn.Conv2d(16, 16, 1, bias=False),
            lateral2=nn.Conv2d(32, 32, 1, bias=False),
            lateral3=nn.Conv2d(64, 64, 1, bias=False),
            lateral4=nn.Conv2d(128, 128, 1, bias=False),
            down=nn.Upsample(scale_factor=2),
            down1=nn.Conv2d(128, 64, 1, bias=False),
            down2=nn.Conv2d(64, 32, 1, bias=False),
            down3=nn.Conv2d(32, 16, 1, bias=False),
            outconv1=nn.Conv2d(128, 64, kernel_size=3, stride=3, bias=False),
            outconv2=nn.Conv2d(64, 32, kernel_size=3, stride=3, bias=False),
            outconv3=nn.Conv2d(32, 16, kernel_size=3, stride=3, bias=False),
            outconv4=nn.Conv2d(16, 8, kernel_size=3, stride=3, bias=False),
            act=nn.ReLU()
        ))
        self.n3 = self.n2

    def forward(self, x1, x2, x3):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3

        x1 = n1.act(n1.conv1(x1))
        c1 = x1.clone()
        # print(x1.shape)
        skip = n1.skip1(n1.pool(x1.clone()))
        x1 = n1.act(n1.conv2(x1))
        x1 = n1.act(n1.conv3(x1))
        x1 = n1.act(n1.conv3(x1) + skip)
        c2 = x1.clone()
        # print(x1.shape)
        skip = n1.skip2(n1.pool(x1.clone()))
        x1 = n1.act(n1.conv4(x1))
        x1 = n1.act(n1.conv5(x1))
        x1 = n1.act(n1.conv5(x1) + skip)
        c3 = x1.clone()
        # print(x1.shape)
        skip = n1.skip3(n1.pool(x1.clone()))
        x1 = n1.act(n1.conv6(x1))
        x1 = n1.act(n1.conv7(x1))
        x1 = n1.act(n1.conv7(x1) + skip)
        c4 = x1.clone()
        # print(x1.shape)
        m4 = n1.act(n1.lateral4(c4))
        p4 = n1.act(n1.outconv1(m4))
        m3 = n1.down1(n1.down(m4)) + n1.act(n1.lateral3(c3))
        p3 = n1.act(n1.outconv2(m3))
        m2 = n1.down2(n1.down(m3)) + n1.act(n1.lateral2(c2))
        p2 = n1.act(n1.outconv3(m2))
        m1 = n1.down3(n1.down(m2)) + n1.act(n1.lateral1(c1))
        p1 = n1.act(n1.outconv4(m1))
        # print(p4.shape, p3.shape, p2.shape, p1.shape)
        p4 = p4.reshape(-1, 256)
        p3 = p3.reshape(-1, 640)
        p2 = p2.reshape(-1, 1280)
        p1 = p1.reshape(-1, 2688)
        x1 = torch.cat((p4, p3, p2, p1), dim=1)

        x2 = n2.act(n2.conv1(x2))
        c1 = x2.clone()
        # print(x2.shape)
        skip = n2.skip1(n2.pool(x2.clone()))
        x2 = n2.act(n2.conv2(x2))
        x2 = n2.act(n2.conv3(x2))
        x2 = n2.act(n2.conv3(x2) + skip)
        c2 = x2.clone()
        # print(x2.shape)
        skip = n2.skip2(n2.pool(x2.clone()))
        x2 = n2.act(n2.conv4(x2))
        x2 = n2.act(n2.conv5(x2))
        x2 = n2.act(n2.conv5(x2) + skip)
        c3 = x2.clone()
        # print(x2.shape)
        skip = n2.skip3(n2.pool(x2.clone()))
        x2 = n2.act(n2.conv6(x2))
        x2 = n2.act(n2.conv7(x2))
        x2 = n2.act(n2.conv7(x2) + skip)
        c4 = x2.clone()
        # print(x2.shape)
        m4 = n2.act(n2.lateral4(c4))
        p4 = n2.act(n2.outconv1(m4))
        m3 = n2.down1(n2.down(m4)) + n2.act(n2.lateral3(c3))
        p3 = n2.act(n2.outconv2(m3))
        m2 = n2.down2(n2.down(m3)) + n2.act(n2.lateral2(c2))
        p2 = n2.act(n2.outconv3(m2))
        m1 = n2.down3(n2.down(m2)) + n2.act(n2.lateral1(c1))
        p1 = n2.act(n2.outconv4(m1))
        # print(p4.shape, p3.shape, p2.shape, p1.shape)
        p4 = p4.reshape(-1, 128)
        p3 = p3.reshape(-1, 256)
        p2 = p2.reshape(-1, 640)
        p1 = p1.reshape(-1, 1280)
        x2 = torch.cat((p4, p3, p2, p1), dim=1)

        x3 = n3.act(n3.conv1(x3))
        c1 = x3.clone()
        # print(x3.shape)
        skip = n3.skip1(n3.pool(x3.clone()))
        x3 = n3.act(n3.conv2(x3))
        x3 = n3.act(n3.conv3(x3))
        x3 = n3.act(n3.conv3(x3) + skip)
        c2 = x3.clone()
        # print(x3.shape)
        skip = n3.skip2(n3.pool(x3.clone()))
        x3 = n3.act(n3.conv4(x3))
        x3 = n3.act(n3.conv5(x3))
        x3 = n3.act(n3.conv5(x3) + skip)
        c3 = x3.clone()
        # print(x3.shape)
        skip = n3.skip3(n3.pool(x3.clone()))
        x3 = n3.act(n3.conv6(x3))
        x3 = n3.act(n3.conv7(x3))
        x3 = n3.act(n3.conv7(x3) + skip)
        c4 = x3.clone()
        # print(x3.shape)
        m4 = n3.act(n3.lateral4(c4))
        p4 = n3.act(n3.outconv1(m4))
        m3 = n3.down1(n3.down(m4)) + n3.act(n3.lateral3(c3))
        p3 = n3.act(n3.outconv2(m3))
        m2 = n3.down2(n3.down(m3)) + n3.act(n3.lateral2(c2))
        p2 = n3.act(n3.outconv3(m2))
        m1 = n3.down3(n3.down(m2)) + n3.act(n3.lateral1(c1))
        p1 = n3.act(n3.outconv4(m1))
        # print(p4.shape, p3.shape, p2.shape, p1.shape)
        p4 = p4.reshape(-1, 128)
        p3 = p3.reshape(-1, 256)
        p2 = p2.reshape(-1, 640)
        p1 = p1.reshape(-1, 1280)
        x3 = torch.cat((p4, p3, p2, p1), dim=1)

        x = torch.cat((x1, x2, x3), dim=1)
        # TODO Use transformer for decoding
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
