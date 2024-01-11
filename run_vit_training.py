from transformers import get_linear_schedule_with_warmup
import torch
import tqdm
import os
from utils import load_seeds, ViTLoRA, load_vit_data


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seeds = load_seeds()

    for task in ['mnist3', 'cifar10']:
        num_epochs = 15 if task == 'mnist3' else 30     # Train models for 15 epochs for MNIST, 30 for CIFAR

        for num_per_class in [10, 20, 30, 40, 50, 60]:
            # Load the preprocessed data
            trainset, _ = load_vit_data(task, num_per_class)

            for seed in seeds:
                # Set up the save path if it does not exist yet
                save_path = f'{os.getcwd()}/models/vit/{task}_{num_per_class}pc/{seed}/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                def collate_fn(examples):
                    pixel_values = torch.stack([example["pixel_values"] for example in examples])
                    labels = torch.tensor([example["label"] for example in examples])
                    return {"pixel_values": pixel_values, "labels": labels}
                
                # Set the seed and dataloader
                torch.manual_seed(seed)
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, collate_fn=collate_fn, shuffle=True)

                # Load the LoRA model 
                model = ViTLoRA(device=device)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
                lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer=optimizer,
                    num_warmup_steps=0,
                    num_training_steps=(len(train_loader) * num_epochs),
                )

                # Train model and save last 5 checkpoints
                model.train()
                for epoch in range(num_epochs):
                    for batch in tqdm.tqdm(train_loader):
                        inputs = batch['pixel_values']
                        labels = batch['labels']
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs, labels=labels)
                        loss = outputs.loss.mean()
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                    if epoch > (num_epochs-5):
                        torch.save(model.state_dict(), os.path.join(save_path, f'ckpt_{epoch}.pth'))


if __name__=="__main__":
    main()
