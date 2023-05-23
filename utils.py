import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision.datasets import MNIST, CIFAR10
from transformers import AutoModelForImageClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    ToTensor,
)
from types import List


def load_seeds():
    return torch.load(f'{os.getcwd()}/random_seeds.pt')


class NetBW(nn.Module):
    def __init__(self):
        super(NetBW, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(in_features=64*5*5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x, output_hidden_states=False):
        x = self.pool(nn.functional.gelu(self.conv1(x)))
        x = self.pool(nn.functional.gelu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x_latent = nn.functional.gelu(self.fc1(x))
        x = self.fc2(x_latent)
        if output_hidden_states:
            return (x, x_latent)
        return x
    

class NetRGB(nn.Module):
    def __init__(self):
        super(NetRGB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.fc1 = nn.Linear(in_features=64*5*5, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x, output_hidden_states=False):
        x = self.pool(nn.functional.gelu(self.conv1(x)))
        x = self.pool(nn.functional.gelu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x_latent = nn.functional.gelu(self.fc1(x))
        x = self.fc2(x_latent)
        if output_hidden_states:
            return (x, x_latent)
        return x
    

class MNISTWithIdx(MNIST):
    def __getitem__(self, index):
        img, target = super(MNISTWithIdx, self).__getitem__(index)
        return img, target, index
    

class CIFAR10WithIdx(CIFAR10):
    def __getitem__(self, index):
        img, target = super(CIFAR10WithIdx, self).__getitem__(index)
        return img, target, index


def train_model(model, train_loader, optimizer, criterion, num_epochs, save_path=None, loo_idx=None):
    """Model training with option to leave one out by zero-ing out the loss. Saves last 5 checkpoints."""
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        for batch in train_loader:
            inputs, labels, indices = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if loo_idx is not None:
                if loo_idx in indices:
                    loss[torch.where(torch.isin(indices, loo_idx))] = 0     #Remove from loss contribution
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        if save_path is not None:
            if (epoch+1) > (num_epochs - 5):
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': running_loss / len(train_loader),
                    }, os.path.join(save_path, f'ckpt_epoch_{epoch}.pth'))
    return model


def test_model(model, test_loader, criterion):
    """Runs the CNN model and returns the test loss, accuracy and predictions."""
    model.eval()
    correct = 0
    total = 0
    test_loss = []
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, _ = batch
            outputs = model(inputs)
            _, predicted = torch.max(torch.nn.functional.softmax(outputs), axis=1)
            total += labels.size(0)
            predictions.append(predicted.cpu().numpy())
            correct += (predicted == labels).sum().item()
            test_loss.append(criterion(outputs, labels).cpu().numpy())

    accuracy = 100 * correct / total

    # Concatenate the predicted values into a single numpy array
    predictions = np.concatenate(predictions)
    test_loss = np.concatenate(test_loss)

    return test_loss, accuracy, predictions


def compute_gradient(model, criterion, instance):
    """Computes parameter gradient of the model for a given input."""
    input, label = instance[0], instance[1]
    
    # Forward pass to compute the loss
    outputs = model(input)
    loss = criterion(outputs, label)
    
    model.zero_grad()
    
    # Extract the gradients of the inputs tensor
    gradient_tuple = torch.autograd.grad(outputs=loss, 
                                   inputs=[param for _, param
                                        in model.named_parameters()
                                        if param.requires_grad])
    
    return gradient_tuple


def ViTLoRA(device):
    """Loads the ViT model as a peft model with LoRA."""
    peft_config = LoraConfig(r=16,
                            lora_alpha=16,
                            target_modules=["query", "value"],
                            lora_dropout=0.1,
                            bias="none",
                            modules_to_save=["classifier"],
                            )

    model = AutoModelForImageClassification.from_pretrained(
                'google/vit-base-patch16-224-in21k',
                num_labels=10,
                ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
            )

    model = get_peft_model(model, peft_config)
    model = model.to(device)
    return model


def test_vit(data_loader, device, model):
    """Run the ViT model on the data in dataloader and return accuracy, loss and predictions."""
    loss = []
    preds = []
    num_correct = 0
    total =0
    for batch in tqdm(data_loader):
        inputs = batch['pixel_values']
        labels = batch['labels']
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
        logits = outputs.logits
        pred = logits.argmax(dim=-1)
        tmp_correct = (pred == labels).sum().item()
        num_correct += tmp_correct
        total += len(labels)
        loss.extend(list(outputs.loss.cpu().numpy()))
        preds.extend(list(pred.cpu().numpy()))
    accuracy = num_correct * 1. / total
    return accuracy, loss, preds


def load_vit_data(task, num_per_class):
    """Load the image data preprocessed by AutoImageProcessor for the ViT."""
    dataset_name = 'mnist' if 'mnist' in task else 'cifar10'    # Define the dataset name as in HuggingFace Datasets

    # Load the image processor and data transforms
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    data_transforms = Compose(
        [
            RandomResizedCrop(image_processor.size["height"]),
            ToTensor(),
            normalize,
        ]
    )

    def preprocess_data(example_batch):
        """Apply data_transforms across a batch."""
        example_batch["pixel_values"] = [data_transforms(image.convert("RGB")) for image in example_batch["image"]]
        return example_batch
    
    # Load the training set and subselect it
    trainset = load_dataset(dataset_name, split='train')
    if task == 'cifar10':
        trainset = trainset.rename_column('img', 'image')
    trainset = trainset.add_column('idx', range(len(trainset)))
    trainset.set_transform(preprocess_data)
    train_idx = load_subset_indices(f'{os.getcwd()}/../data/{task}/train_subset_{num_per_class}pc.txt')
    trainset = trainset.select((idx for idx in range(len(trainset))
                                if idx in train_idx))
    
    # Load the test set and subselect it
    testset = load_dataset(dataset_name, split='test')
    if task == 'cifar10':
        testset = testset.rename_column('img', 'image')
    testset = testset.add_column('idx', range(len(testset)))
    testset.set_transform(preprocess_data)
    test_idx = load_subset_indices(f'{os.getcwd()}/../data/{task}/test_subset.txt')
    testset = testset.select((idx for idx in range(len(testset))
                                if idx in test_idx))
    
    return trainset, testset


def load_subset_indices(idx_filepath):
    """Reads indices defined in a text file at idx_filepath."""
    with open(idx_filepath, 'r') as f:
        indices = f.readlines()
    indices = [int(idx.strip()) for idx in indices]
    return indices


def load_attribution_types():
    return ['loo', 'ats', 'if', 'gd', 'gc']


def get_expected_tau_per_z(expected_attributions, test_idx):
    """Organise the expected attributions."""
    seeds = load_seeds()
    # Reordering the attributions into respective dataframes
    expected_tau_per_z = {}
    for z_test_idx in test_idx:
        df = pd.DataFrame()
        for seed in seeds:
            df[seed] = expected_attributions[seed][f'z_test_{z_test_idx}']
        expected_tau_per_z[f'z_test_{z_test_idx}'] = df
    return expected_tau_per_z


def load_expected_tda_swa(num_ckpts: int,
                               experiment: str,
                               tau: List[str] = ['loo', 'ats', 'if', 'gd', 'gc']):
    """Loading the expected attribution of type tau across the last num_ckpts checkpoints."""
    seeds = load_seeds()
    expected_tda = {}
    model_name, task, num_per_class = experiment.split('_')
    max_ckpt = 15 if 'mnist3'==task else 30
    ckpts = range(max_ckpt-num_ckpts, max_ckpt) 

    for seed in seeds:
        cumulative_attribution=None
        for num_ckpt in ckpts:
            attribution = pd.read_csv(f'{os.getcwd()}/tda_scores/{model_name}/{tau}/{task}_{num_per_class}pc/{seed}/attribution_ckpt_{num_ckpt}.csv', index_col=False)
            cumulative_attribution = attribution if cumulative_attribution is None else cumulative_attribution + attribution
        expected_tda[seed] = cumulative_attribution/len(ckpts)
    return expected_tda


def get_mu_and_sigma(tau: str, 
                     num_ckpts: int, 
                     experiment: str, 
                     test_idx: List[int]):
    """Compute mean and standard deviation across random seeds and checkpoints for all train-test pairs."""
    expected_attributions = load_expected_tda_swa(num_ckpts=num_ckpts, 
                                                    experiment=experiment,
                                                    tau=tau)
    expected_tau_per_z = get_expected_tau_per_z(expected_attributions=expected_attributions,
                                                test_idx=test_idx)
    all_means = pd.DataFrame()
    all_stds = pd.DataFrame()
    for z_test_idx in test_idx:
        means = expected_tau_per_z[f'z_test_{z_test_idx}'].mean(axis=1)
        all_means[z_test_idx] = means
        stds = expected_tau_per_z[f'z_test_{z_test_idx}'].std(axis=1)
        all_stds[z_test_idx] = stds
    return all_means, all_stds

