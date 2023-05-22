import torch
from torchvision import transforms
import numpy as np
import os
from utils import (MNISTWithIdx, CIFAR10WithIdx, load_subset_indices)


def get_subset_indices(dataset, num_per_class): 
    """Randomly chooses num_per_class indices from dataset to make a balanced subset and returns a list of indices."""
    # Get the indices of each class in the dataset
    indices = {}
    for i in range(len(dataset)):
        _, label,_ = dataset[i]
        if label not in indices:
            indices[label] = []
        indices[label].append(i)

    # Select a balanced subset of the dataset
    subset_indices = []
    for label in indices:
        subset_indices += np.random.choice(indices[label], num_per_class, replace=False).tolist()
    return subset_indices


def main():
    # Download the data from torchvision
    trainset_mnist = MNISTWithIdx(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    testset_mnist = MNISTWithIdx(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    transform_cifar = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset_cifar = CIFAR10WithIdx(root='./data', train=True, transform=transform_cifar, download=True)
    testset_cifar = CIFAR10WithIdx(root='./data', train=False, transform=transform_cifar, download=True)

    for task, datasets in zip(['mnist3', 'cifar10'], [(trainset_mnist, testset_mnist), (trainset_cifar, testset_cifar)]):
        trainset, testset = datasets
        for num_per_class in [10, 20, 50]:
            # Create the balanced subset dataset (loading same indices as used in our study)
            train_indices = load_subset_indices(f'{os.getcwd()}/data/{task}/train_subset_{num_per_class}pc.txt')
            test_indices = load_subset_indices(f'{os.getcwd()}/data/{task}/test_subset.txt')

            # !! If you would like to define new subsets, uncomment this:
            # train_indices = get_subset_indices(trainset, num_per_class)
            # test_indices = get_subset_indices(testset, num_per_class)

            train_subset = torch.utils.data.Subset(trainset, train_indices)
            test_subset = torch.utils.data.Subset(testset, test_indices)

            # Save
            torch.save(train_subset, f'{os.getcwd()}/data/{task}/train_subset_{num_per_class}pc.pt')
            torch.save(test_subset, f'{os.getcwd()}/data/{task}/test_subset.pt')



if __name__=='__main__':
    main()