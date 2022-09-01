import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from network import Network


class TrainingData(Dataset):
    def __init__(self):
        data = np.loadtxt("train.csv", dtype="float32", delimiter=",")
        self.x = torch.from_numpy(data[:, 1:])
        self.x = torch.reshape(self.x, (-1, 96, 128, 3))
        self.x = torch.swapaxes(self.x, 1, 3)
        self.x = torch.swapaxes(self.x, 2, 3)
        self.y = torch.from_numpy(data[:, [0]])
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
num_epochs = 10
batch_size = 100

# import torch.nn.functional as f
# conv1 = nn.Conv2d(3, 5, 5, stride=2)
# pool = torch.nn.MaxPool2d(2, 2)
# conv2 = nn.Conv2d(5, 10, 5, stride=2)
#
# images, _ = dataset[0]
# images = images.view(-1, 96, 128, 3)
# images = torch.swapaxes(images, 1, 3)
# images = torch.swapaxes(images, 2, 3)
#
# x = f.pool(f.relu(conv1(images)))
# x = f.pool(f.relu(conv2(x)))
# print(x.shape)

model = Network().to(device)
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

        print(f"Epoch {epoch}, Step {step}, Loss: {loss.item() : .4f}")

# Test
with torch.no_grad():
    for (images, labels) in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        print(loss)

torch.save(model.state_dict(), "model.pth")
