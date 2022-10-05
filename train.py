import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from network import VSNet


class TrainingData(Dataset):
    def __init__(self):
        data = np.loadtxt("vstrainingdata/vs_train_clean.csv", dtype="float32", delimiter=",", max_rows=None)

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
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

device = torch.device("cuda")

model = VSNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

norm1 = torch.nn.BatchNorm2d(3)
norm1 = norm1.to(device)
for epoch in range(1000):
    if (epoch + 1) % 25 == 0:
        torch.save(model.state_dict(), f"models/vs{epoch + 1}.pth")
    for step, (x1, x2, x3, labels) in enumerate(train_dataloader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        # labels = torch.zeros(labels.shape)
        labels = labels.to(device)
        x1 = norm1(x1)
        x2 = norm1(x2)
        x3 = norm1(x3)

        outputs = model(x1, x2, x3)
        train_loss = criterion(outputs, labels)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            for x1, x2, x3, labels in test_dataloader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)
                labels = labels.to(device)
                x1 = norm1(x1)
                x2 = norm1(x2)
                x3 = norm1(x3)

                outputs = model(x1, x2, x3)
                test_loss = criterion(outputs, labels)

        print(f"Epoch {epoch+1}, Step {step+1}, "
              f"Train loss:{train_loss.item() : .4f}, Test loss:{test_loss.item() : .4f}")
    # scheduler.step()

# Test
with torch.no_grad():
    for x1, x2, x3, labels in test_dataloader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        labels = labels.to(device)

        outputs = model(x1, x2, x3)
        loss = criterion(outputs, labels)

        print(loss)
