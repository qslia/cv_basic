import torch
from torchvision import datasets, transforms

train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

trainset = datasets.CIFAR10(
    root='/home/q/task/cv_basic/dataset', train=True, download=True, transform=train_transforms)
testset = datasets.CIFAR10(
    root='/home/q/task/cv_basic/dataset', train=False, download=True, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1024, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=512, shuffle=False, num_workers=8)
