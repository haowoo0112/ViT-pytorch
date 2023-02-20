from torchvision import transforms, datasets
import torchvision as tv
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def get_loader():
    transform_train = transforms.Compose([
        #transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    trainset = datasets.CIFAR10(root="./data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
    testset = datasets.CIFAR10(root="./data",
                                train=False,
                                download=True,
                                transform=transform_test)

    train_sampler = RandomSampler(trainset)
    test_sampler = SequentialSampler(testset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=1,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=4,
                             pin_memory=True)
    
    return train_loader, test_loader

def imshow(imgs):
    """show image from tensor"""

    npimg = tv.utils.make_grid(imgs).numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()