import torch
import numpy as np
import os
from utils import get_mu_and_sigma, load_attribution_types
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def save_corr_matrix(experiment, corr_matrix, save_name):
    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(4,4))

    # Create a colormap
    cmap = sns.diverging_palette(h_neg=197, 
                                 h_pos=24, 
                                 as_cmap=True, 
                                 center='light')

    # Plot the correlation matrix
    sns.heatmap(corr_matrix.round(2), vmin=-1, vmax=1, cmap=cmap, annot=True, annot_kws={"fontsize":14}, fmt=".2f",
                square=True, cbar=False, ax=ax)

    # Set the title and labels
    labels = ['LOO', 'ATS', 'IF', 'GD', 'GC']
    ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize=14)
    ax.set_yticklabels(labels, rotation=0, fontsize=14)

    # Show the plot
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'{os.getcwd()}/results/{experiment}/{save_name}', dpi=300)


def main():
    taus = load_attribution_types()
    num_ckpts=5         # As TDA scores are most stable with max number of checkpoints, we set this to 5.
    experiments = ['cnn_mnist3_10pc', 'cnn_mnist3_20pc']    # Add experiments to this list that you want to analyse

    for experiment in experiments:
        task = experiment.split('_')[1]
        test_subset = torch.load(f'{os.getcwd()}/data/{task}/test_subset.pt')
        test_idx = [idx for _,_,idx in test_subset]

        mus = {}
        sigmas = {}
        ps ={}
        for tau in taus:
            # Compute mean, standard deviation and p-value
            mus[tau], sigmas[tau] = get_mu_and_sigma(tau=tau, num_ckpts=num_ckpts, experiment=experiment, test_idx=test_idx) 
            mus[tau] = mus[tau].values.flatten()
            sigmas[tau] = sigmas[tau].values.flatten()
            ps[tau] = mus[tau]/sigmas[tau]

        # Compute Pearson correlation
        pearson_corr_mu = np.corrcoef(np.stack(mus.values()), rowvar=True)
        pearson_corr_sigma = np.corrcoef(np.stack(sigmas.values()), rowvar=True)
        pearson_corr_p = np.corrcoef(np.stack(ps.values()), rowvar=True)

        # Compute Spearman correlation
        spearman_corr_mu, _ = spearmanr(np.stack(mus.values()), axis=1)
        spearman_corr_sigma, _ = spearmanr(np.stack(sigmas.values()), axis=1)
        spearman_corr_p, _ = spearmanr(np.stack(ps.values()), axis=1)

        # Save correlation matrices 
        save_corr_matrix(experiment=experiment,
                        corr_matrix=spearman_corr_mu,
                        save_name='spearman_corr_mu.pdf')
        
        save_corr_matrix(experiment=experiment,
                        corr_matrix=spearman_corr_sigma,
                        save_name='spearman_corr_sigma.pdf')
        
        save_corr_matrix(experiment=experiment,
                        corr_matrix=spearman_corr_p,
                        save_name='spearman_corr_p.pdf')
        
        save_corr_matrix(experiment=experiment,
                        corr_matrix=pearson_corr_mu,
                        save_name='pearson_corr_mu.pdf')
        
        save_corr_matrix(experiment=experiment,
                        corr_matrix=pearson_corr_sigma,
                        save_name='pearson_corr_sigma.pdf')
        
        save_corr_matrix(experiment=experiment,
                        corr_matrix=pearson_corr_p,
                        save_name='pearson_corr_p.pdf')


if __name__=="__main__":
    main()