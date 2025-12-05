from torchvision import transforms

MNIST_TRANSFORMS = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
