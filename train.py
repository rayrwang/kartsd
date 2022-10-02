import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from network import VSNet


class TrainingData(Dataset):
    def __init__(self):
        data = np.loadtxt("vstrainingdata/vs_train_rough.csv", dtype="float32", delimiter=",", max_rows=None)

        self.x1 = torch.from_numpy(data[:, :147456])
        self.x1 = torch.reshape(self.x1, (-1, 192, 256, 3))
        self.x1 = self.x1[:, 50:, :, :]
        self.x1 = torch.swapaxes(self.x1, 1, 3)
        self.x1 = torch.swapaxes(self.x1, 2, 3)

        self.x2 = torch.from_numpy(data[:, 147456:223488])
        self.x2 = torch.reshape(self.x2, (-1, 144, 176, 3))
        self.x2 = self.x2[:, 85:, :, :]
        self.x2 = torch.swapaxes(self.x2, 1, 3)
        self.x2 = torch.swapaxes(self.x2, 2, 3)

        self.x3 = torch.from_numpy(data[:, 223488:299520])
        self.x3 = torch.reshape(self.x3, (-1, 144, 176, 3))
        self.x3 = self.x3[:, 85:, :, :]
        self.x3 = torch.swapaxes(self.x3, 1, 3)
        self.x3 = torch.swapaxes(self.x3, 2, 3)

        self.y = torch.from_numpy(data[:, 299520:])

        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x1[index], self.x2[index], self.x3[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = TrainingData()
train_size = round(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, test_size))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

device = torch.device("cuda")

model = VSNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for step, (x1, x2, x3, labels) in enumerate(train_dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        labels = labels.to(device)

        outputs = model(x1, x2, x3)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Step {step}, Loss:{loss.item() : .4f}")

# Test
with torch.no_grad():
    for step, (x1, x2, x3, labels) in enumerate(test_dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        labels = labels.to(device)

        outputs = model(x1, x2, x3)
        loss = criterion(outputs, labels)

        print(loss)

torch.save(model.state_dict(), "models/vs.pth")
