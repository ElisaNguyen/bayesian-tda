import torch
import os
import pandas as pd
from scipy import stats
from utils import (load_attribution_types, get_mu_and_sigma)


def main():
    tau_types = load_attribution_types()
    experiment = 'cnn_mnist3_10pc'  # <model>_<task>_<num_per_class>pc

    # Check if the path to save to already exists or not
    if not os.path.exists(f'{os.getcwd()}/results/{experiment}/'):
        os.makedirs(f'{os.getcwd()}/results/{experiment}/')

    # Load the test subset and its ids
    test_subset = torch.load(f'{os.getcwd()}/data/{experiment.split("_")[1]}/test_subset.pt')
    test_idx = [idx for _,_,idx in test_subset]

    ckpts = range(1,6)
    
    # Set up result dataframe for mean p-values.
    df_res = pd.DataFrame()
    df_res['num_ckpts'] = ckpts
    for tau in tau_types:
        mean_p_values = []
        for num_ckpts in ckpts:
            # Compute the means and standard deviations of each train-test pair across random seeds and num_ckpts checkpoints.
            all_means, all_stds = get_mu_and_sigma(tau=tau, 
                                                   num_ckpts=num_ckpts, 
                                                   experiment=experiment, 
                                                   test_idx=test_idx) 

            # Compute z-score and p-value
            z_scores = all_means/all_stds
            p_values = pd.DataFrame(stats.norm.sf(z_scores.abs())*2, columns=all_means.columns)

            # Save to results folder
            p_values.to_csv(f'{os.getcwd()}/results/{experiment}/pvalues_{tau}_across_{num_ckpts}_ckpts.csv', index=False)
            
            # Compute mean p-value of the experiment
            mean_p_value = p_values.values.mean()
            mean_p_values.append(mean_p_value)
        df_res[tau] = mean_p_values
    df_res.to_csv(f'{os.getcwd()}/results/{experiment}/mean_p_values.csv', index=False)


if __name__=="__main__":
    main()