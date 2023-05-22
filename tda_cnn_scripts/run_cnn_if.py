import torch
import os
from torch.utils.data import DataLoader
from nn_influence_utils import compute_influences
from utils import (NetBW, NetRGB, load_seeds)
import pandas as pd
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)

    train_dataset = torch.load(f'{os.getcwd()}/../data/{args.task}/train_subset_{args.num_per_class}pc.pt')
    test_dataset = torch.load(f'{os.getcwd()}/../data/{args.task}/test_subset.pt')
    colnames = [f'z_test_{idx}' for _,_,idx in test_dataset]
    colnames.insert(0, 'train_idx')
    batch_train_data_loader = DataLoader(train_dataset, batch_size=8)
    instance_train_data_loader=DataLoader(train_dataset, batch_size=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seeds = load_seeds()
    seed = seeds[args.seed_id]

    # Hyperparameters of s_test estimation
    s_test_num_samples= min(len(train_dataset), 1000) 
    s_test_damp=5e-3 
    s_test_scale=1e4 
    s_test_iterations = 1

    for num_ckpt in ckpts:
        save_path = f"{os.getcwd()}/../tda_scores/if/{args.task}_{args.num_per_class}pc/{seed}/attribution_ckpt_{num_ckpt}.csv"
        s_test_path = f'{os.getcwd()}/../tda_scores/if/{args.task}_{args.num_per_class}pc/{seed}/'
        precomputed_s_tests = torch.load(f'{s_test_path}/s_tests_ckpt_{num_ckpt}.pt')

        model = NetRGB() if train_dataset[0][0].shape[0]==3 else NetBW()
        ckpt = torch.load(f'{os.getcwd()}/../models/cnn/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        df_if = pd.DataFrame()
        df_if['train_idx'] = [idx for _,_,idx in train_dataset]
        for z_test in test_dataset: 
            z_test_idx = z_test[2]
            precomputed_s_test = precomputed_s_tests[z_test_idx]
            # Inluences is dict {train_sample_index: influence} will be of size num_training_samples
            influences = compute_influences(n_gpu=1,
                                            device=device,
                                            model=model,
                                            test_inputs=z_test, 
                                            batch_train_data_loader=batch_train_data_loader, 
                                            instance_train_data_loader=instance_train_data_loader,
                                            s_test_num_samples = s_test_num_samples,
                                            s_test_damp = s_test_damp, 
                                            s_test_scale=s_test_scale,
                                            s_test_iterations=s_test_iterations,
                                            precomputed_s_test=precomputed_s_test, 
                                            )

            # Save influences
            df_if[f"z_test_{z_test_idx}"] = influences.values()
            df_if.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()