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
        trainset.data = trainset.train_data[:num_train, :, :]
        trainset.targets = trainset.train_labels[:num_train]

        # Validation set
        valset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform) # (@Mo): we don't need this, can use above
        valset.data = valset.train_data[num_train:, :, :]
        valset.targets = valset.train_labels[num_train:]

        # Test set
        testset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
        testset.data = testset.test_data

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
        trainset.data = trainset.train_data[:num_train, :, :]
        trainset.labels = trainset.train_labels[:num_train]
        # Validation set
        valset = datasets.FashionMNIST(root='./data/fashion', train=True, download=True, transform=transform)
        valset.data = valset.train_data[num_train:, :, :]
        valset.labels = valset.train_labels[num_train:]
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


def load_cifar10(batch_size, num_train=45000, val_split=True, augmentation=False, subset=[-1, -1, -1],
                 only_split_train=False):
    train_transforms = []
    test_transforms = []

    if augmentation:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     # How about we compute these dynamically, not hard-coded? Then we can handle CIFAR-10 and CIFAR-100 etc.
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transforms.append(normalize)
    test_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    if val_split:
        # num_train = 45000  # Will split training set into 45,000 training and 5,000 validation images
        # Train set
        original_trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True,
                                             transform=train_transform)
        trainset = original_trainset
        trainset.data = trainset.train_data[:num_train, :, :, :]
        trainset.labels = trainset.train_labels[:num_train]
        # Validation set
        valset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=test_transform)
        valset.data = valset.train_data[num_train:, :, :, :]
        valset.labels = valset.train_labels[num_train:]
        # Test set
        testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)

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
                                      num_workers=0)  # 45,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    num_workers=0)  # 5,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=2)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=2)  # 10,000 images

        return train_dataloader, None, test_dataloader


def load_cifar100(batch_size, num_train=45000, val_split=True, augmentation=False, subset=[-1, -1, -1]):
    train_transforms = []
    test_transforms = []

    if augmentation:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    train_transforms.append(normalize)
    test_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    if val_split:
        # Train set
        trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)
        trainset.data = trainset.train_data[:num_train, :, :, :]
        trainset.labels = trainset.train_labels[:num_train]
        # Validation set
        valset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=test_transform)
        valset.data = valset.train_data[num_train:, :, :, :]
        valset.labels = valset.train_labels[num_train:]
        # Test set
        testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])
        if subset[1] != -1:
            valset = getSubset(valset, subset[1])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 45,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)  # 5,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        trainset = datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=train_transform)
        testset = datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=test_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)  # 10,000 images

        return train_dataloader, None, test_dataloader


def load_ham(batch_size, val_split=True, augmentation=False, subset=[-1, -1, -1]):
    train_transforms = []
    test_transforms = []

    if augmentation:
        train_transforms.append(transforms.RandomHorizontalFlip())
        train_transforms.append(transforms.RandomVerticalFlip())
        train_transforms.append(transforms.CenterCrop(256))
        train_transforms.append(transforms.RandomCrop(224))

    train_transforms.append(transforms.ToTensor())

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)

    base_skin_dir = os.path.join('..', 'data/HAM')

    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                         for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))}

    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
    tile_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
    tile_df['path'] = tile_df['image_id'].map(imageid_path_dict.get)
    tile_df['cell_type'] = tile_df['dx'].map(lesion_type_dict.get)
    tile_df['cell_type_idx'] = pd.Categorical(tile_df['cell_type']).codes
    train_df, test_df = train_test_split(tile_df, test_size=0.1)
    train_df = train_df.reset_index()

    if val_split:
        validation_df, test_df = train_test_split(test_df, test_size=0.5)
        validation_df = validation_df.reset_index()
        test_df = test_df.reset_index()
        # Train set
        trainset = HAM_dataset(train_df, transform=train_transform)
        # Validation set
        valset = HAM_dataset(validation_df, transform=train_transform)
        # Test set
        testset = HAM_dataset(test_df, transform=train_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])
        if subset[1] != -1:
            valset = getSubset(valset, subset[1])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=0)  # 45,000 images
        val_dataloader = DataLoader(valset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                    num_workers=0)  # 5,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=0)  # 10,000 images

        return train_dataloader, val_dataloader, test_dataloader
    else:
        test_df = test_df.reset_index()
        # Train set
        trainset = HAM_dataset(train_df, transform=train_transform)
        # Validation set
        testset = HAM_dataset(test_df, transform=train_transform)

        if subset[0] != -1:
            trainset = getSubset(trainset, subset[0])
        if subset[2] != -1:
            testset = getSubset(testset, subset[2])

        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                      num_workers=2)  # 50,000 images
        test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                     num_workers=2)  # 10,000 images

        return train_dataloader, None, test_dataloader
