# A Bayesian Perspective On Training Data Attribution
This repository contains the Python code for the experiments of NeurIPS'23 submission #12645.

Training data attribution (TDA) techniques find influential training data for the model's prediction on the test data of interest. They approximate the impact of down- or up-weighting a particular training sample. While conceptually useful, they are hardly applicable in practice, particularly because of their sensitivity to different model initialisation. In this paper, we introduce a Bayesian perspective on the TDA task, where the learned model is treated as a Bayesian posterior and the TDA estimates as random variables. From this novel viewpoint, we observe that the influence of an individual training sample is often overshadowed by the noise stemming from model initialisation and SGD batch composition. Based on this observation, we argue that TDA can only be reliably used for explaining model predictions that are consistently influenced by certain training data, independent of other noise factors. Our experiments demonstrate the rarity of such noise-independent training-test data pairs but confirm their existence. We recommend that future researchers and practitioners trust TDA estimates only in such cases. Further, we find a disagreement between ground truth and estimated TDA distributions and encourage future work to study this gap. 

------------------------------
## Reproducing the experiments

### Requirements

The main dependencies are:

- `python==3.10.4`
- `torch`
- `torchvision`
- `transformers`
- `datasets`
- `numpy`
- `pandas`
- `scikit-learn`

A `conda_env.yml` is provided that details the packages required for reproducing the experiments. 
We conducted the experiments using this environment on a Nvidia 2080ti GPU.

### Data

We subsample MNIST and CIFAR10, and provide the indices of the datasets used in our experiments in `data/subset_indices`. 
To reproduce the dataset, run `run_subset_generation.py`. 

### Models

To train the CNN models, run `run_cnn_training.py`. 
To finetune the ViT model, run `run_vit_finetuning.py`.

These scripts train the respective model 10 times on the seeds specified in `random_seeds.pt`, which we also use in the paper. This corresponds to sampling a model $\theta$ from the posterior $p(\theta|\mathcal{D})$ using Deep Ensembling. 
The checkpoints after the last 5 epochs are saved to `models/`. This corresponds to sampling a model $\theta$ from the posterior $p(\theta|\mathcal{D})$ similar to stochastic weight averaging. 

### Experiments
#### Computing the 
#### p-values
