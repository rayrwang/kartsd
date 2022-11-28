import torch

from networks import VSNet

device = torch.device("cpu")
model = VSNet().to(device)
model.load_state_dict(torch.load("models/vs.pth", map_location=device))
model.eval()

ex0 = torch.rand(1, 3, 480, 640)
ex1 = torch.rand(1, 3, 480, 640)
ex2 = torch.rand(1, 3, 480, 640)
ex3 = torch.rand(1, 3, 480, 640)
ex4 = torch.rand(1, 3, 480, 640)

traced = torch.jit.trace(model, (ex0, ex1, ex2, ex3, ex4))
traced.save("cinfer/cmodels/cvs.pt")
