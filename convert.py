import torch

from networks import VSNet

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/vs.pth", map_location=device))
model.eval()

ex1 = torch.rand(1, 3, 142, 256)
ex2 = torch.rand(1, 3, 59, 176)
ex3 = torch.rand(1, 3, 59, 176)

traced = torch.jit.trace(model, (ex1, ex2, ex3))
traced.save("cinfer/models/cvs.pt")
