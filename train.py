import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from networks import VSNet


class TrainingData(Dataset):
    def __init__(self):
        data = np.load("vstrainingdata/vs_train_clean.npy")
        data = data.astype("float32")

        for i in range(5):
            xn = torch.from_numpy(data[:, i*640*480*3:(i+1)*640*480*3])
            xn = torch.reshape(xn, (-1, 480, 640, 3))
            xn = torch.swapaxes(xn, 1, 3)
            xn = torch.swapaxes(xn, 2, 3)
            setattr(self, f"x{i}", xn)

        self.y = torch.from_numpy(data[:, 5*640*480*3:])

        self.n_samples = data.shape[0]

    def __getitem__(self, index):
        return self.x0[index], self.x1[index], self.x2[index], self.x3[index], self.x4[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = TrainingData()
train_size = round(0.8*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, (train_size, test_size))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

device = torch.device("cuda")

model = VSNet().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(100_000):
    model.train()
    for step, (x0, x1, x2, x3, x4, y) in enumerate(train_dataloader):
        x0 = x0.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        y = y.to(device)

        yh = model(x0, x1, x2, x3, x4)
        train_loss = criterion(yh, y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        try:
            print(f"Epoch {epoch + 1}, Step {step + 1}, "
                  f"Train loss:{train_loss.item() : .4f}, Test loss:{test_loss.item() : .4f}")
        except NameError:
            print(f"Epoch {epoch + 1}, Step {step + 1}, "
                  f"Train loss:{train_loss.item() : .4f}, Test loss: Not defined yet")

    with torch.no_grad():
        model.eval()
        for step, (x0, x1, x2, x3, x4, y) in enumerate(test_dataloader):
            x0 = x0.to(device)
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            y = y.to(device)

            yh = model(x0, x1, x2, x3, x4)
            test_loss = criterion(yh, y)

    if (epoch + 1) % 25 == 0:
        torch.save(model.state_dict(), f"models/test/vs{epoch + 1}.pth")
