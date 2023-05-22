import torch
import torch.nn as nn
import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from utils import (NetBW, NetRGB, load_seeds, compute_gradient)


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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seeds = load_seeds()
    seed = seeds[args.seed_id]

    criterion = nn.CrossEntropyLoss(reduction='none')

    for num_ckpt in ckpts:
        model = NetRGB() if train_dataset[0][0].shape[0]==3 else NetBW()
        ckpt = torch.load(f'{os.getcwd()}/../models/cnn/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval() 

        save_path_gd = f"{os.getcwd()}/../tda_scores/gd/{args.task}_{args.num_per_class}pc/{seed}/attribution_ckpt_{num_ckpt}.csv"
        save_path_gc = f"{os.getcwd()}/../tda_scores/gc/{args.task}_{args.num_per_class}pc/{seed}/attribution_ckpt_{num_ckpt}.csv"

        if not os.path.exists(os.path.split(save_path_gd)[0]):
            os.makedirs(os.path.split(save_path_gd)[0])
        if not os.path.exists(os.path.split(save_path_gc)[0]):
            os.makedirs(os.path.split(save_path_gc)[0])

        df_gd = pd.DataFrame()
        df_gc = pd.DataFrame()
        df_gd['train_idx'] = [idx for _,_,idx in train_dataset]
        df_gc['train_idx'] = [idx for _,_,idx in train_dataset]

        test_instance_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        train_instance_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

        for z_test in tqdm(test_instance_loader):
            z_test_idx = z_test[2].cpu().item()
            gd = []
            gc = []
            
            grad_z_test = compute_gradient(model=model,
                                           criterion=criterion,
                                           instance=z_test)
            flat_grad_z_test = torch.concat([layer_grad.flatten() for layer_grad in grad_z_test])
            for z_train in train_instance_loader:
                grad_z_train = compute_gradient(model=model,
                                           criterion=criterion,
                                           instance=z_train)
                flat_grad_z_train = torch.concat([layer_grad.flatten() for layer_grad in grad_z_train])
                
                grad_dot = torch.dot(flat_grad_z_test, flat_grad_z_train)
                gd.append(grad_dot.item())

                grad_cos = nn.functional.cosine_similarity(flat_grad_z_test, flat_grad_z_train, dim=0)
                gc.append(grad_cos.item())
            df_gd[f'z_test_{z_test_idx}'] = gd
            df_gc[f'z_test_{z_test_idx}'] = gc

        df_gd.to_csv(save_path_gd, index=False)
        df_gc.to_csv(save_path_gc, index=False)
            

if __name__=="__main__":
    main()