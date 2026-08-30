
from torchvision import datasets

datasets.MNIST(root="data", train=True, download=True)
datasets.MNIST(root="data", train=False, download=True)

print("MNIST downloaded to data/")
