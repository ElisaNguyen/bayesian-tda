import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader 
from utils import (NetBW, NetRGB, train_model, load_seeds)


def main():
    seeds = load_seeds()
    for task in ['mnist3', 'cifar10']:
        for num_per_class in [10, 20, 50]:
            train_dataset = torch.load(f'{os.getcwd()}/data/{task}/train_subset_{num_per_class}.pt')
            num_epochs = 15 if task == 'mnist3' else 30     # Train models for 15 epochs for MNIST, 30 for CIFAR

            for seed in seeds:
                torch.manual_seed(seed)

                model = NetRGB() if train_dataset[0][0].shape[0]==3 else NetBW()
                criterion = nn.CrossEntropyLoss(reduction='none')
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

                # Train and record the last 5 checkpoints
                train_model(model, 
                            train_loader=train_loader, 
                            optimizer=optimizer,
                            criterion=criterion,
                            num_epochs=num_epochs,
                            save_path=f'{os.getcwd()}/models/{task}/{seed}')


if __name__=="__main__":
    main()