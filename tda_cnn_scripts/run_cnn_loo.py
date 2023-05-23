import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils import (NetBW, NetRGB, train_model, load_seeds, test_model)
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    # Load datasets and variables needed for the computation
    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)
    train_dataset = torch.load(f'{os.getcwd()}/../data/{args.task}/train_subset_{args.num_per_class}pc.pt')
    test_dataset = torch.load(f'{os.getcwd()}/../data/{args.task}/test_subset.pt')
    colnames = [f'z_test_{idx}' for _,_,idx in test_dataset]
    colnames.insert(0, 'train_idx')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    seeds = load_seeds()
    seed = seeds[args.seed_id]
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Set up save path for saving results
    save_path = f"{os.getcwd()}/../tda_scores/cnn/loo/{args.task}_{args.num_per_class}pc/{seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for num_ckpt in ckpts:
        # Set up dataframe for results
        df_loo = pd.DataFrame(columns=colnames)
        df_loo['train_idx'] = [idx for _,_,idx in train_dataset]
        for _,_, z_train_idx in train_dataset:
            # Load the model and get the initial loss values
            model = NetRGB() if train_dataset[0][0].shape[0]==3 else NetBW()
            ckpt = torch.load(f'{os.getcwd()}/../models/cnn/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            test_loss, _, _ = test_model(model=model,
                                        test_loader=test_loader,
                                        criterion=criterion)

            
            model.train()      # Set model to train mode for retraining 
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)    # Same parameters as training

            # Set the random seed and load the training set
            torch.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Train with z_train_idx removed and record the checkpoints
            model = train_model(model, 
                                train_loader=train_loader, 
                                optimizer=optimizer,
                                criterion=criterion,
                                num_epochs=num_epochs,
                                loo_idx=z_train_idx)
            
            # Run the loo model on the test set
            model.eval()
            loo_loss, _, _ = test_model(model=model,
                                        test_loader=test_loader,
                                        criterion=criterion)
            
            # Record the loss change 
            delta_loss = loo_loss - test_loss
            row_idx = np.where(df_loo['train_idx']==z_train_idx)[0][0]
            df_loo.loc[row_idx, colnames[1]:] = delta_loss
            
            # Save 
            df_loo.to_csv(f"{save_path}/attribution_ckpt_{num_ckpt}.csv", index=False)


if __name__=="__main__":
    main()