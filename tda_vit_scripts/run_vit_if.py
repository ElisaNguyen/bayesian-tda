
import torch
import os
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from nn_influence_utils_vit import compute_influences, compute_gradients
from utils import load_seeds, ViTLoRA, load_vit_data
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    seeds = load_seeds()
    seed = seeds[args.seed_id]

    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load datasets 
    train_dataset, test_dataset = load_vit_data(args.task, args.num_per_class)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        idx = torch.tensor([example["idx"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels, "idx": idx}
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    instance_train_data_loader=DataLoader(train_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    # Hyperparameters of s_test estimation
    s_test_num_samples= min(len(train_loader), 1000) 
    s_test_damp=5e-3 
    s_test_scale=1e4 
    s_test_iterations = 1

    s_test_path = f'{os.getcwd()}/../tda_scores/vit/if/{args.task}/{seed}/'

    for num_ckpt in ckpts: 
        if os.path.exists(f'{s_test_path}/s_tests_ckpt_{num_ckpt}.pt'):
            precomputed_s_tests = torch.load(f'{s_test_path}/s_tests_ckpt_{num_ckpt}.pt')
        else:
            precomputed_s_tests = None

        # Load the model
        model = ViTLoRA(device=device)
        state_dict = torch.load(f'{os.getcwd()}/../models/vit/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
        model.load_state_dict(state_dict)

        save_path = f'{os.getcwd()}/../tda_scores/vit/if/{args.task}_{args.num_per_class}pc/{seed}/attribution_ckpt_{num_ckpt}.csv'
        if os.path.exists(save_path):
            df_attribution = pd.read_csv(save_path, index_col=False)
            if df_attribution.shape == (len(train_dataset), len(test_dataset)+1):
                continue
            else:
                finished_z_test = [eval(colname.split('_')[-1]) for colname in df_attribution.columns[1:]]
        else:
            df_attribution = pd.DataFrame()
            df_attribution['train_idx'] = [instance['idx'] for instance in train_dataset]
            finished_z_test = []

        # Compute the z_train gradients
        grads_zj = {}
        for train_inputs in tqdm(instance_train_data_loader):
            grad_zj = compute_gradients(
                n_gpu=1,
                device=device,
                model=model,
                inputs=train_inputs,
                params_filter=None,
                weight_decay=None,
                weight_decay_ignores=None)
            grads_zj[train_inputs['idx'].item()] = grad_zj

        # Compute influences
        for z_test in test_loader:
            z_test_idx = z_test['idx'].item()
            if z_test_idx in finished_z_test:
                continue
            if precomputed_s_tests is not None:
                precomputed_s_test = precomputed_s_tests[z_test_idx]
            else:
                precomputed_s_test = None

            influences = compute_influences(n_gpu=1,
                                            device=device,
                                            model=model,
                                            test_inputs=z_test,
                                            batch_train_data_loader=train_loader,
                                            instance_train_data_loader=instance_train_data_loader,
                                            s_test_num_samples=s_test_num_samples,
                                            s_test_iterations=s_test_iterations,
                                            s_test_scale=s_test_scale,
                                            s_test_damp=s_test_damp,
                                            precomputed_s_test=precomputed_s_test,
                                            precomputed_grad_zjs=grads_zj)
            # save influences
            df_attribution[f"z_test_{z_test_idx}"] = influences.values()
            df_attribution.to_csv(save_path, index=False)
            torch.cuda.empty_cache()


if __name__=="__main__":
    main()