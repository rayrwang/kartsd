import math

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

        self.ln_1 = nn.LayerNorm(128)
        self.attn = nn.MultiheadAttention(128, 1, batch_first=True)
        self.ln_2 = nn.LayerNorm(128)
        self.ff1 = nn.Linear(128, 128)
        self.ff2 = nn.Linear(128, 128)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8192, 500)
        self.fc2 = nn.Linear(500, 12120)

        # Positional encoding
        self.pe = torch.zeros(1000, 64, 128, device="cuda")
        for pos in range(64):
            for e_dim in range(128):
                i = e_dim // 2
                if e_dim % 2 == 0:
                    self.pe[:, pos, e_dim] = math.sin(pos / 10000 ** (2 * i / 128))
                else:
                    self.pe[:, pos, e_dim] = math.cos(pos / 10000 ** (2 * i / 128))

    def forward(self, x1, x2, x3):
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3

        x1 = n1.act(n1.conv1(x1))
        x1 = n1.act(n1.conv2(x1))
        x1 = n1.act(n1.conv3(x1))
        x1 = n1.act(n1.conv4(x1))
        # print(x1.shape)
        x1 = x1.reshape(-1, 128, 28)
        x1 = torch.transpose(x1, 1, 2)

        x2 = n2.act(n2.conv1(x2))
        x2 = n2.act(n2.conv2(x2))
        x2 = n2.act(n2.conv3(x2))
        # print(x2.shape)
        x2 = x2.reshape(-1, 128, 18)
        x2 = torch.transpose(x2, 1, 2)

        x3 = n3.act(n3.conv1(x3))
        x3 = n3.act(n3.conv2(x3))
        x3 = n3.act(n3.conv3(x3))
        # print(x3.shape)
        x3 = x3.reshape(-1, 128, 18)
        x3 = torch.transpose(x3, 1, 2)

        x = torch.cat((x1, x2, x3), 1)  # x shape = -1 64 128 (batch, seq length, embed dims)
        # print(x.shape)

        # Transformer
        skip = x.clone()
        self.ln_1(x)
        x = self.attn(self.pe[:x.shape[0], :, :], x, x, need_weights=False)[0] + skip
        skip = x.clone()
        x = self.ln_2(x)
        x = F.relu(self.ff1(x))
        x = F.relu(self.ff2(x) + skip)

        # Decoding head
        x = x.reshape(-1, 8192)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x
