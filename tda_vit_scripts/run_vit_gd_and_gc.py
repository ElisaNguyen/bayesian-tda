import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import (load_seeds, load_vit_data, ViTLoRA)
from nn_influence_utils_vit import compute_gradients
from argparse import ArgumentParser
import os
import pandas as pd
        

def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seeds = load_seeds()
    seed = seeds[args.seed_id]

    # Load the dataset
    train_dataset, test_dataset = load_vit_data(args.task, args.num_per_class)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        idx = torch.tensor([example["idx"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels, "idx": idx}
    train_loader = DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)

    for num_ckpt in ckpts:
        save_path_dp = f"{os.getcwd()}/../tda_scores/vit/gd/{args.task}_{args.num_per_class}pc/{seed}/"
        if not os.path.exists(save_path_dp):
            os.makedirs(save_path_dp)
        save_path_cs = f"{os.getcwd()}/../tda_scores/vit/gc/{args.task}_{args.num_per_class}pc/{seed}/"
        if not os.path.exists(save_path_cs):
            os.makedirs(save_path_cs)
        
        df_dp = pd.DataFrame()
        df_dp['train_idx'] = [instance['idx'] for instance in train_dataset]
        df_cos = pd.DataFrame()
        df_cos['train_idx'] = [instance['idx'] for instance in train_dataset]

        model = ViTLoRA(device=device)
        state_dict = torch.load(f'{os.getcwd()}/../models/vit/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
        model.load_state_dict(state_dict)

        # get the gradients
        for z_test in test_loader:
            z_test_idx = z_test['idx'].item()
            
            dp_attribution = []
            cs_attribution = []

            grad_z_test = compute_gradients(device=device,
                                            model=model,
                                            inputs=z_test,
                                            params_filter=None)
            flat_grad_z_test = torch.concat([layer_grad.flatten() for layer_grad in grad_z_test])
            for z_train in train_loader: 
                grad_z_train = compute_gradients(device=device,
                                                model=model,
                                                inputs=z_train,
                                                params_filter=None)
                flat_grad_z_train = torch.concat([layer_grad.flatten() for layer_grad in grad_z_train])

                # Compute dot product
                grad_dot = torch.dot(flat_grad_z_test, flat_grad_z_train)
                dp_attribution.append(grad_dot.item())

                # Comput cosine similarity
                grad_cos = nn.functional.cosine_similarity(flat_grad_z_test, flat_grad_z_train, dim=0)
                cs_attribution.append(grad_cos.item())
                
                
            df_dp[f'z_test_{z_test_idx}'] = dp_attribution
            df_cos[f'z_test_{z_test_idx}'] = cs_attribution
        df_dp.to_csv(os.path.join(save_path_dp, f"attribution_ckpt_{num_ckpt}.csv"), index=False)
        df_cos.to_csv(os.path.join(save_path_cs, f"attribution_ckpt_{num_ckpt}.csv"), index=False)
            

if __name__=="__main__":
    main()