# A Bayesian Approach To Analysing Training Data Attribution In Deep Learning

#### Elisa Nguyen, Minjoon Seo, Seong Joon Oh

Training data attribution (TDA) techniques find influential training data for the model's prediction on the test data of interest. They approximate the impact of down- or up-weighting a particular training sample. While conceptually useful, they are hardly applicable to deep models in practice, particularly because of their sensitivity to different model initialisation. In this paper, we introduce a Bayesian perspective on the TDA task, where the learned model is treated as a Bayesian posterior and the TDA estimates as random variables. From this novel viewpoint, we observe that the influence of an individual training sample is often overshadowed by the noise stemming from model initialisation and SGD batch composition. Based on this observation, we argue that TDA can only be reliably used for explaining deep model predictions that are consistently influenced by certain training data, independent of other noise factors. Our experiments demonstrate the rarity of such noise-independent training-test data pairs but confirm their existence. We recommend that future researchers and practitioners trust TDA estimates only in such cases. Further, we find a disagreement between ground truth and estimated TDA distributions and encourage future work to study this gap.

#### [Link to paper](https://arxiv.org/abs/2305.19765)

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

To train the CNN models, run `run_cnn_training.py`. This trains two layer CNNs. If you want to train three layer CNNs, update this script with the respective classes.  
To finetune the ViT model with LoRA, run `run_vit_finetuning.py`.

These scripts train the respective model 10 times on the seeds specified in `random_seeds.pt`, which we also use in the paper. This corresponds to sampling a model $\theta$ from the posterior $p(\theta|\mathcal{D})$ using Deep Ensembling. 
The checkpoints after the last 5 epochs are saved to `models/`. This corresponds to sampling a model $\theta$ from the posterior $p(\theta|\mathcal{D})$ similar to stochastic weight averaging. 

This is done for all datasets, if you wish to run it on a specific one, change it directly in the script. 

### Experiments
In the paper, we conduct hypothesis testing of the signal-to-noise ratio in TDA scores and report the p-value as an indicator of the statistical significance of the estimated scores. Additionally, we inspect the Pearson and Spearman correlations of the TDA scores of different methods to find out how well they correspond to each other. Below are instructions on how to reproduce these analyses. 

#### Step 1: Computing the TDA scores
We test 5 different TDA methods. We provide the scripts in the folders `tda_cnn_scripts` and `tda_vit_scripts` for computing the TDA scores of across the ensemble of models for the CNN and ViT respectively. 

Each of the scripts should be called with the following parameters: `python <script_name> --task <string> --num_per_class <int> --seed_id <int>`. 

- `--task` specifies the task of the experiment, i.e. either `mnist3` or `cifar10`.
- `--num_per_class` is an integer $\in$ {10, 20, 50} that refers to how many samples per class the model was trained on.
- `--seed_id` is an integer that specifies the seed from the `random_seeds.pt` file. This parameter is used for parallel processing, in case multiple GPUs are available. 

For computing influence functions, we use the code provided by the [FastIF repository](https://github.com/salesforce/fast-influence-functions). Beware to compute the HVP s_test before computing the influence function. 

Please note that this step may take a while, depending on the size of the model. 

#### Step 2: Computing p-values
After TDA scores are computed, we can analyse the reliability of the scores by the p-values. To compute p-values across the TDA scores computed for each train-test pair of each sample $\theta$, run `run_compute_pvalues.py`. This will generate CSV files of the p-values in a new `results/<experiment>` folder.

Specify the experiment to compute p-values for as `<model>_<task>_<num_per_class>pc`, e.g. `cnn_mnist3_10pc`. 

#### Step 3: Computing correlations
We provide the script to compute the Pearson and Spearman correlation between the mean, standard deviation and p-values computed across the different samples $\theta$ in `run_correlation_analysis.py`. This will save the correlation matrices in the `results/<experiment>` folder.

----------
## Contact
For any problem with implementation or bug, please contact <a href="mailto:elisa.nguyen@live.de" target="_blank">Elisa Nguyen</a>
.

## How to cite
```
@inproceedings{nguyen2023bayesiantda,
    title = {A Bayesian Perspective On Training Data Attribution},
    author = {Nguyen, Elisa and Seo, Minjoon and Oh, Seong Joon},
    year = {2023},
    booktitle = {Conference on Neural Information Processing Systems},
}
```
