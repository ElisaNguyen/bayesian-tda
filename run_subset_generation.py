import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd
from utils import (NetBW, NetRGB, test_model, train_model, load_seeds, MNISTWithIdx, CIFAR10WithIdx)


def get_subset_indices(subset, num_per_class: int): 
    # Get the indices of each class in the dataset
    indices = {}
    for i in range(len(subset)):
        _, label,_ = subset[i]
        if label not in indices:
            indices[label] = []
        indices[label].append(i)

    # Select a balanced subset of the dataset
    subset_indices = []
    for label in indices:
        subset_indices += np.random.choice(indices[label], num_per_class, replace=False).tolist()
    return subset_indices


def load_indices(idx_filepath):
    with open(idx_filepath, 'r') as f:
        indices = f.readlines()
    indices = [int(idx.strip()) for idx in indices]
    return indices


def main():
    # Download the data from torchvision
    transform_cifar = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_cifar = CIFAR10WithIdx(root='./data', train=True, transform=transform_cifar, download=True)
    testset_cifar = CIFAR10WithIdx(root='./data', train=False, transform=transform_cifar, download=True)

    # Create the balanced subset dataset (loading same indices as used in our study)
    train_indices = load_indices(f'{os.getcwd()}/data/cifar10/train_subset_10pc.txt')
    test_indices = load_indices(f'{os.getcwd()}/data/cifar10/test_subset.txt')

    # !! If you would like to define new subsets, uncomment this:
    # train_indices = get_subset_indices(trainset_cifar, 10)
    # test_indices = get_subset_indices(testset_cifar, 10)

    train_subset = torch.utils.data.Subset(trainset_cifar, train_indices)
    test_subset = torch.utils.data.Subset(testset_cifar, test_indices)

    # Save
    torch.save(train_subset, f'{os.getcwd()}/data/cifar10/train_subset_10pc.pt')
    torch.save(test_subset, f'{os.getcwd()}/data/cifar10/test_subset.pt')



if __name__=='__main__':
    main()