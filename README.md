# A Bayesian Perspective On Training Data Attribution
This repository contains the Python code for the experiments of NeurIPS'23 submission #12645.

Training data attribution (TDA) techniques find influential training data for the model's prediction on the test data of interest. They approximate the impact of down- or up-weighting a particular training sample. While conceptually useful, they are hardly applicable in practice, particularly because of their sensitivity to different model initialisation. In this paper, we introduce a Bayesian perspective on the TDA task, where the learned model is treated as a Bayesian posterior and the TDA estimates as random variables. From this novel viewpoint, we observe that the influence of an individual training sample is often overshadowed by the noise stemming from model initialisation and SGD batch composition. Based on this observation, we argue that TDA can only be reliably used for explaining model predictions that are consistently influenced by certain training data, independent of other noise factors. Our experiments demonstrate the rarity of such noise-independent training-test data pairs but confirm their existence. We recommend that future researchers and practitioners trust TDA estimates only in such cases. Further, we find a disagreement between ground truth and estimated TDA distributions and encourage future work to study this gap. 

------------------------------
## Reproducing the experiments

### Requirements

The main dependencies are:

- `python==3.10.4`
- `torch==2.0.0`
- `torchvision==0.15.0`
- `transformers==4.28.1`
- `datasets==2.12.0`
- `numpy==1.23.5`
- `pandas==1.5.2`
- `scikit-learn==1.2.2`
- `seaborn==0.12.2`

A `req.txt` is provided that details the packages required for reproducing the experiments. To install the same conda environment, use `$ conda create --name <env> --file req.txt` 
We conducted the experiments using this environment on a Nvidia 2080ti GPU.

### Data

We subsample MNIST and CIFAR10, and provide the indices of the datasets used in our experiments in `data/subset_indices`. 
To reproduce the dataset, run `run_subset_generation.py`. 

### Models

To train the CNN models, run `run_cnn_training.py`. 
To finetune the ViT model with LoRA, run `run_vit_finetuning.py`.

These scripts train the respective model 10 times on the seeds specified in `random_seeds.pt`, which we also use in the paper. This corresponds to sampling a model $\theta$ from the posterior $p(\theta|\mathcal{D})$ using Deep Ensembling. 
The checkpoints after the last 5 epochs are saved to `models/`. This corresponds to sampling a model $\theta$ from the posterior $p(\theta|\mathcal{D})$ similar to stochastic weight averaging. 

This is done for all datasets, if you wish to run it on a specific one, change it directly in the script. 

### Experiments
In the paper, we conduct hypothesis testing of the signal-to-noise ratio in TDA scores and report the p-value as an indicator of the statistical significance of the estimated scores. Additionally, we inspect the Pearson and Spearman correlations of the TDA scores of different methods to find out how well they correspond to each other. Below are instructions on how to reproduce these analyses. 

#### Step 1: Computing the TDA scores
We test 5 different TDA methods. To compute the TDA scores of across the ensemble of models, run the following:

- For LOO: `python run_loo.py --experiment {str} --seed_id {int}`
- For ATS: `python run_ats.py --experiment {str} --seed_id {int}`
- For IF: `python run_if.py --experiment {str} --seed_id {int}`
- For GD and GC: `python run_gd_and_gc.py --experiment {str} --seed_id {int}`

The parameter `--experiment` specifies the name of the experiment. This should be the same as the model folder of the model to compute TDA for. 
The parameter `--seed_id` is an integer that specifies the seed from the `random_seeds.pt` file. This parameter is used for parallel processing, in case multiple GPUs are available. 

#### Step 2: Computing p-values
After TDA scores are computed, we can compute the 

#### Step 3: Computing correlations
xxxx
