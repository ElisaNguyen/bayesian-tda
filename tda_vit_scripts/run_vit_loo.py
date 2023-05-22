from transformers import get_linear_schedule_with_warmup
import numpy as np
import torch
import tqdm
import os
from argparse import ArgumentParser
from utils import load_seeds, ViTLoRA, test_vit, load_vit_data
import pandas as pd
from torch.utils.data import DataLoader


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seeds = load_seeds()
    seed = seeds[args.seed_id]

    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)

    save_path = f"{os.getcwd()}/../tda_scores/vit/loo/{args.task}_{args.num_per_class}pc/{seed}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Load data
    trainset, testset = load_vit_data(args.task, args.num_per_class)
    
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        idx = torch.tensor([example["idx"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels, "idx": idx}

    torch.manual_seed(seed)
    train_loader = DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64, collate_fn=collate_fn, shuffle=False)
    
    colnames = [f"z_test_{instance['idx']}" for instance in testset]
    colnames.insert(0, 'train_idx')

    for num_ckpt in ckpts: 
        df_loo = pd.DataFrame(columns=colnames)
        df_loo['train_idx'] = [instance['idx'] for instance in trainset]

        # Load the model trained with whole dataset
        model = ViTLoRA(device=device)
        state_dict = torch.load(f'{os.getcwd()}/../models/vit/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
        model.load_state_dict(state_dict)

        # Get loss and preds of the model trained on whole dataset
        _, full_loss, _ = test_vit(data_loader=test_loader,
                                     device=device,
                                     model=model)

        ##### Retrain the model without the training instance
        for instance in trainset:
            # Load the pretrained model
            model = ViTLoRA(device=device)

            # Set up training
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
            lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=(len(train_loader) * num_epochs),
            )
            model.train()

            # Train
            for _ in range(num_epochs):
                for batch in tqdm.tqdm(train_loader):
                    inputs = batch['pixel_values']
                    labels = batch['labels']
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    # If the train instance to remove is in this batch, zero out the loss
                    if instance['idx'] in batch['idx']:
                        idx_to_remove = torch.where(instance['idx'] == batch['idx'])[0].item()
                        loss[idx_to_remove] = 0
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            # Get loss and preds of the model trained on dataset with one removed
            model.eval()
            _, loo_loss, _ = test_vit(data_loader=test_loader,
                                        device=device,
                                        model=model)

            # Record the loss change 
            delta_loss = np.array(loo_loss) - np.array(full_loss)
            row_idx = np.where(df_loo['train_idx']==instance['idx'])[0][0]
            df_loo.loc[row_idx, colnames[1]:] = delta_loss
        # Save
        df_loo.to_csv(f"{save_path}/attribution_ckpt_{num_ckpt}.csv", index=False)
      


if __name__=="__main__":
    main()
