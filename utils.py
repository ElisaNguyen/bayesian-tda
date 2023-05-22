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
    model.eval()
    correct = 0
    total = 0
    test_loss = []
    predictions = []
    softmax_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels, _ = batch
            outputs = model(inputs)
            probs, predicted = torch.max(torch.nn.functional.softmax(outputs), axis=1)
            total += labels.size(0)
            predictions.append(predicted.cpu().numpy())
            softmax_probs.append(probs.cpu().numpy())
            correct += (predicted == labels).sum().item()
            test_loss.append(criterion(outputs, labels).cpu().numpy())

    accuracy = 100 * correct / total

    # Concatenate the predicted values into a single numpy array
    predictions = np.concatenate(predictions)
    softmax_probs = np.concatenate(softmax_probs)
    test_loss = np.concatenate(test_loss)

    # print('Test Loss: {:.4f}, Accuracy: {:.2f}%'.format(test_loss.mean(), accuracy))
    return test_loss, accuracy, predictions, softmax_probs


def compute_gradient(model, criterion, instance):
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


def compute_grad_grad(model, criterion, instance):
    input, label = instance[0], instance[1]
    
    # Forward pass to compute the loss
    outputs = model(input)
    loss = criterion(outputs, label)
    
    # Backward pass to compute gradients
    model.zero_grad()
    # loss.backward()
    
    # # Extract the gradients of the inputs tensor
    # gradient = input.grad
    gradient_tuple = torch.autograd.grad(outputs=loss, 
                                   inputs=[
                                    param for _, param
                                    in model.named_parameters()
                                    if param.requires_grad], 
                                   grad_outputs=loss, 
                                   create_graph=True)

    model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(outputs=gradient_tuple, 
                                    inputs=[
                                    param for _, param
                                    in model.named_parameters()
                                    if param.requires_grad], 
                                    grad_outputs=gradient_tuple)

    return grad_grad_tuple


def dot_product_loss_gradients(model, criterion, data1, data2):
    grad1 = compute_gradient(model, criterion, data1)
    grad2 = compute_gradient(model, criterion, data2)

    # Compute dot product of gradients
    # dot_product = torch.dot(grad1.squeeze().flatten(), grad2.squeeze().flatten()) # this is only for one
    dot_product = torch.sum(grad1*grad2, axis=(1,2,3))
    
    return dot_product


def cos_similarity_loss_gradient(model, criterion, data1, data2):
    grad1 = compute_gradient(model, criterion, data1)
    grad2 = compute_gradient(model, criterion, data2)

    # Compute cosine similarity
    flat_dim = grad1.squeeze().flatten().size(0)
    cos_sim = nn.functional.cosine_similarity(grad1.reshape([1, flat_dim]), grad2.reshape([grad2.shape[0], flat_dim]), dim=1)
    return cos_sim


def compute_attribution(model, criterion, z_test_set, train_dataset, save_path, attribution_method):
    if not os.path.exists(os.path.split(save_path)[0]):
        os.makedirs(os.path.split(save_path)[0])
    df_attribution = pd.DataFrame()
    df_attribution['train_idx'] = [idx for _,_,idx in train_dataset]
    test_instance_loader = torch.utils.data.DataLoader(z_test_set, batch_size=1, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
    for z_test in tqdm(test_instance_loader):
        z_test_idx = z_test[2].cpu().item()
        attribution = []
        for z_train_batch in train_loader:
            attribution.extend(attribution_method(model=model,
                                                    criterion=criterion,
                                                    data1=z_test,
                                                    data2=z_train_batch).cpu())
        df_attribution[f'z_test_{z_test_idx}'] = [x.item() for x in attribution]
    df_attribution.to_csv(save_path, index=False)


def ViTLoRA(device):
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
    train_idx = load_subset_indices(f'{os.getcwd()}/data/{task}/train_subset_{num_per_class}pc.txt')
    trainset = trainset.select((idx for idx in range(len(trainset))
                                if idx in train_idx))
    
    # Load the test set and subselect it
    testset = load_dataset(dataset_name, split='test')
    if task == 'cifar10':
        testset = testset.rename_column('img', 'image')
    testset = testset.add_column('idx', range(len(testset)))
    testset.set_transform(preprocess_data)
    test_idx = load_subset_indices(f'{os.getcwd()}/data/{task}/test_subset.txt')
    testset = testset.select((idx for idx in range(len(testset))
                                if idx in test_idx))
    
    return trainset, testset


def load_subset_indices(idx_filepath):
    """Reads indices defined in a text file at idx_filepath."""
    with open(idx_filepath, 'r') as f:
        indices = f.readlines()
    indices = [int(idx.strip()) for idx in indices]
    return indices