import os
import ipdb
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Local imports
from HAM_dataset import HAM_dataset


def getSubset(data, size):
    return Subset(data, np.random.randint(0, high=len(data) - 1, size=size))


def load_boston(batch_size, val_split=True, subset=[-1, -1, -1], num_train=50000):
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import torch

    boston = load_boston()
    X, y = (boston.data, boston.target)

    X_use, X_test, y_use, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_use, y_use, test_size=0.1, random_state=0)

    from torch.utils.data import TensorDataset, DataLoader
    traindataset = TensorDataset(torch.from_numpy(X_train).clone().float(), torch.from_numpy(y_train).clone().float())
    valdataset = TensorDataset(torch.from_numpy(X_val).clone().float(), torch.from_numpy(y_val).clone().float())
    testdataset = TensorDataset(torch.from_numpy(X_test).clone().float(), torch.from_numpy(y_test).clone().float())

    train_dataloader = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=valdataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_mnist(batch_size, val_split=True, subset=[-1, -1, -1], num_train=50000, only_split_train=False):
    transformations = [transforms.ToTensor()]
    transformations.append(transforms.Normalize((0.1307,), (0.3081,)))
    transform = transforms.Compose(transformations)

    if val_split:
        # num_train = 50000  # Will split training set into 50,000 training and 10,000 validation images
        # Train set
        original_trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        trainset = original_trainset
        trainset.train_data = trainset.train_data[:num_train, :, :]
        trainset.train_labels = trainset.train_labels[:num_train]

        # Validation set
        valset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        valset.train_data = valset.train_data[num_train:, :, :]
        valset.train_labels = valset.train_labels[num_train:]

        # Test set
        testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        testset.test_data = testset.test_data

        if only_split_train:
            rand_ind = np.random.randint(0, high=len(original_trainset) - 1, size=subset[0] + subset[1])
            if subset[0] != -1:
                trainset = Subset(original_trainset, rand_ind[:subset[0]])
            if subset[2] != -1:
                testset = getSubset(testset, subset[2])
            if subset[1] != -1:
                valset = Subset(original_trainset, rand_ind[subset[0]:subset[0] + subset[1]])
        else:
            if subset[0] != -1:
                trainset = getSubset(trainset, subset[0])
            if subset[2] != -1:
                testset = getSubset(testset, subset[2])
            if subset[1] != -1:
                valset = getSubset(valset, subset[1])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0)  # 50,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    num_workers=0)  # 10,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                     num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, pin_memory=True, shuffle=True,
                                      num_workers=0)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, pin_memory=True, shuffle=False,
                                     num_workers=0)  # 10,000 images
        return train_dataloader, None, test_dataloader


def load_fashion_mnist(batch_size, val_split=True, subset=[-1, -1, -1]):
    transformations = [transforms.ToTensor()]
    transformations.append(transforms.Normalize((0.1307,), (0.3081,)))  # This makes a huge difference!
    transform = transforms.Compose(transformations)

    if val_split:
        num_train = 50000  # Will split training set into 50,000 training and 10,000 validation images
        # Train set
        trainset = datasets.FashionMNIST(root='./data/fashion', train=True, download=True, transform=transform)
        trainset.train_data = trainset.train_data[:num_train, :, :]
        trainset.train_labels = trainset.train_labels[:num_train]
        # Validation set
        valset = datasets.FashionMNIST(root='./data/fashion', train=True, download=True, transform=transform)
        valset.train_data = valset.train_data[num_train:, :, :]
        valset.train_labels = valset.train_labels[num_train:]
        # Test set
        testset = datasets.FashionMNIST(root='./data/fashion', train=False, download=True, transform=transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])
        if subset[1] != -1:
            valset = getSubset(valset, subset[1])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 50,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)  # 10,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.FashionMNIST(root='./data/fashion', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data/fashion', train=False, download=True, transform=transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 60,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)  # 10,000 images
        return train_dataloader, None, test_dataloader
