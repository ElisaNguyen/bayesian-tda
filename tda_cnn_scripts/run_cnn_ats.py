import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils import (NetBW, NetRGB, train_model, load_seeds, test_model)


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)

    seeds = load_seeds()
    seed = seeds[args.seed_id]

    save_path = f"{os.getcwd()}/../tda_scores/cnn/ats/{args.task}_{args.num_per_class}pc/{seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    criterion = nn.CrossEntropyLoss(reduction='none')
    train_dataset = torch.load(f'{os.getcwd()}/../data/{args.task}/train_subset_{args.num_per_class}pc.pt')
    test_dataset = torch.load(f'{os.getcwd()}/../data/{args.task}/test_subset.pt')
    colnames = [f'z_test_{idx}' for _,_,idx in test_dataset]
    colnames.insert(0, 'train_idx')
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for num_ckpt in ckpts:
        df_ats = pd.DataFrame(columns=colnames)
        df_ats['train_idx'] = [idx for _,_,idx in train_dataset]
        for data, label, z_train_idx in train_dataset:
            # Load the model and get the initial loss values
            model = NetRGB() if train_dataset[0][0].shape[0]==3 else NetBW()
            ckpt = torch.load(f'{os.getcwd()}/../models/cnn/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            test_loss, _, _ = test_model(model=model,
                                        test_loader=test_loader,
                                        criterion=criterion)

            # Set model to train mode
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.005)

            # Train for 1 step
            train_instance_loader = DataLoader([[data, label, z_train_idx]], batch_size=1, shuffle=False)
            model = train_model(model, 
                                train_loader=train_instance_loader, 
                                optimizer=optimizer,
                                criterion=criterion,
                                num_epochs=1)
            
            # Run the ATS model on the test set
            model.eval()
            loss, _, _ = test_model(model=model,
                                    test_loader=test_loader,
                                    criterion=criterion)
            
            # Record the loss change 
            delta_loss = loss - test_loss
            row_idx = np.where(df_ats['train_idx']==z_train_idx)[0][0]
            df_ats.loc[row_idx, colnames[1]:] = delta_loss
            
            # Save 
            df_ats.to_csv(f"{save_path}/attribution_ckpt_{num_ckpt}.csv", index=False)


if __name__=="__main__":
    main()