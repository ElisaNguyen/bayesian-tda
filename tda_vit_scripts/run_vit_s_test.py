
import torch
import os
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from nn_influence_utils_vit import compute_s_test
from utils import load_seeds, ViTLoRA, load_vit_data
import warnings
warnings.filterwarnings('ignore')
torch.autograd.set_detect_anomaly(True)


def main():
    parser = ArgumentParser()
    parser.add_argument('--seed_id', type=int, default=0)
    parser.add_argument('--task', type=str, default='mnist3', help='Either mnist3 or cifar10')
    parser.add_argument('--num_per_class', type=int, default=10, help='Number of samples per class that the model was trained on from {10,20,50}')
    args = parser.parse_args()

    # Load variables needed for the computation
    seeds = load_seeds()
    seed = seeds[args.seed_id]
    num_epochs = 15 if 'mnist' in args.task else 30
    ckpts = range(num_epochs-5, num_epochs)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load datasets 
    train_dataset, test_dataset = load_vit_data(args.experiment)
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        idx = torch.tensor([example["idx"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels, "idx": idx}
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    # Hyperparameters of s_test estimation
    s_test_num_samples= min(len(train_loader), 1000) 
    s_test_damp=5e-3 
    s_test_scale=1e4 
    s_test_iterations = 1

    for num_ckpt in ckpts: 
        save_path = f'{os.getcwd()}/../tda_scores/if/{args.task}_{args.num_per_class}/{seed}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        s_tests = {}
        # Load the model
        model = ViTLoRA(device)
        state_dict = torch.load(f'{os.getcwd()}/../models/vit/{args.task}_{args.num_per_class}pc/{seed}/ckpt_epoch_{num_ckpt}.pth')
        model.load_state_dict(state_dict)


        for z_test in test_loader:
            s_test = None
            for _ in range(s_test_iterations):
                _s_test, _ = compute_s_test(
                    n_gpu=1,
                    device=device,
                    model=model,
                    test_inputs=z_test,
                    train_data_loaders=[train_loader],
                    params_filter= None,
                    weight_decay= None,
                    weight_decay_ignores= None,
                    damp=s_test_damp,
                    scale=s_test_scale,
                    num_samples=s_test_num_samples,
                    verbose=False)

                # Sum the values across runs
                if s_test is None:
                    s_test = _s_test
                else:
                    s_test = [
                        a + b for a, b in zip(s_test, _s_test)
                    ]
            # Do the averaging
            s_test = [a / s_test_iterations for a in s_test] # list of tensors
            s_tests[z_test['idx'].item()] = s_test
        
        # Save s_test and history
        torch.save(s_tests, f'{save_path}/s_tests_ckpt_{num_ckpt}.pt')


if __name__=="__main__":
    main()