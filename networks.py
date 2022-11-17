import torch
import torch.nn as nn
import torch.nn.functional as F


class VSNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.bn = nn.BatchNorm2d(3)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(12600, 500)
        self.fc2 = nn.Linear(500, 2*120*101)

        # ResNets
        self.r = nn.ModuleDict(dict(
            act=nn.ReLU(),
            conv1=nn.Conv2d(3, 20, 7, 2),
            conv2=nn.Conv2d(20, 20, 3, 2),
            conv3=nn.Conv2d(20, 20, 3, 1)
        ))

    def forward(self, x0, x1, x2, x3, x4):
        r = self.r

        x = [x0, x1, x2, x3, x4]
        for i in range(5):
            x[i] = self.bn(x[i])
            x[i] = r.act(r.conv1(x[i]))
            x[i] = r.act(r.conv2(x[i]))
            x[i] = r.act(r.conv2(x[i]))
            x[i] = r.act(r.conv2(x[i]))
            x[i] = r.act(r.conv2(x[i]))
            x[i] = r.act(r.conv3(x[i]))
            x[i] = r.act(r.conv3(x[i]))
            # print(x[i].shape)
            x[i] = x[i].reshape(-1, 2520)

        x = torch.cat((x[0], x[1], x[2], x[3], x[4]), dim=1)

        # TODO Use transformer for decoding

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # todo maybe normalize outputs -1 to 1

        return x
