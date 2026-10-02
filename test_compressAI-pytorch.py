import torch
from compressai.zoo import bmshj2018_factorized

print("Loading CompressAI model...")

model = bmshj2018_factorized(quality=1, pretrained=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

x = torch.rand(1, 3, 256, 256).to(device)

with torch.no_grad():
    output = model(x)

print("Device :", device)
print("Input  :", x.shape)
print("Output keys:", output.keys())
print("CompressAI + PyTorch test: OK")