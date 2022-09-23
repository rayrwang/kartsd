import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from network import VSNet


class TrainingData(Dataset):
    def __init__(self):
        # # data = np.loadtxt("center.csv", dtype="float32", delimiter=",")
        # # self.x = torch.from_numpy(data[:, 1:])
        # # self.x = torch.reshape(self.x, (-1, 96, 128, 3))
        # # self.x = torch.swapaxes(self.x, 1, 3)
        # # self.x = torch.swapaxes(self.x, 2, 3)
        # # self.y = torch.from_numpy(data[:, [0]])
        # #
        # # self.n_samples = data.shape[0]
        #
        # left_np = np.loadtxt("left.csv", dtype="float32", delimiter=",")
        # center_np = np.loadtxt("center.csv", dtype="float32", delimiter=",")
        # right_np = np.loadtxt("right.csv", dtype="float32", delimiter=",")
        #
        # left = torch.from_numpy(left_np[:, 1:])
        # center = torch.from_numpy(center_np[:, 1:])
        # right = torch.from_numpy(right_np[:, 1:])
        #
        # self.x = torch.cat((left, center, right))
        #
        # self.x = torch.reshape(self.x, (-1, 96, 128, 3))
        # self.x = torch.swapaxes(self.x, 1, 3)
        # self.x = torch.swapaxes(self.x, 2, 3)
        #
        # left = torch.from_numpy(left_np[:, [0]])
        # center = torch.from_numpy(center_np[:, [0]])
        # right = torch.from_numpy(right_np[:, [0]])
        #
        # left = torch.full(left.shape, -3.5)
        # right = torch.full(right.shape, 0.5)
        #
        # self.y = torch.cat((left, center, right))
        #
        # self.n_samples = self.x.shape[0]

        data = np.loadtxt("vs_train.csv", dtype="float32", delimiter=",")

        self.x = torch.from_numpy(data[:, :36864])
        self.x = torch.reshape(self.x, (-1, 96, 128, 3))
        self.x = torch.swapaxes(self.x, 1, 3)
        self.x = torch.swapaxes(self.x, 2, 3)
        self.y = torch.from_numpy(data[:, 36864:])

        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = TrainingData()
train_size = round(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, test_size))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

device = torch.device("cuda")
batch_size = 100

# import torch.nn.functional as f
# conv1 = nn.Conv2d(3, 5, 5, stride=1)
# pool = torch.nn.MaxPool2d(2, 2)
# conv2 = nn.Conv2d(5, 10, 5, stride=2)
#
# images, _ = dataset[0]
# images = images.reshape(-1, 96, 128, 3)
# images = torch.swapaxes(images, 1, 3)
# images = torch.swapaxes(images, 2, 3)
#
# x = pool(f.relu(conv1(images)))
# x = pool(f.relu(conv2(x)))
# print(x.shape)

model = VSNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for step, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Step {step}, Loss:{loss.item() : .4f}")

# Test
with torch.no_grad():
    for (images, labels) in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        print(loss)

torch.save(model.state_dict(), "models/vs.pth")
