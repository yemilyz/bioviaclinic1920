import os 
import glob
from itertools import permutations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from metric_analysis import *

def get_all_metrics(n_splits=10):
    metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result_{}splits/'.format(n_splits), topdown=False):
        # if 'python' in root:
        #     print(root, 'skipped')
        #     continue
        for name in dirs:     
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                metric = pd.read_csv(metric_file)
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                metric['descriptor'] = descriptor
                if metrics_all.empty:
                    metrics_all = metric
                else:
                    metrics_all =  metrics_all.append(metric)
    new_descriptor = metrics_all["descriptor"].str.rsplit("_", n = 3, expand = True)

    metrics_all['feature'] = new_descriptor[0]
    metrics_all['model'] = new_descriptor[1]
    return metrics_all.reset_index()

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def plot_pareto_front(dim1, dim2, metric_data, pareto_indices, marker_size=1.5):
    metric_data_pareto = metric_data.iloc[pareto_indices]
    # metric_data_pareto.sort_values(dim1,  inplace=True)
    x_all = metric_data[dim1]
    y_all = metric_data[dim2]
    x_pareto = metric_data_pareto[dim1]
    y_pareto = metric_data_pareto[dim2]

    plt.scatter(x_all, y_all, s=marker_size, alpha=0.6)
    plt.scatter(x_pareto, y_pareto, s=marker_size, color='r')
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.title('Pareto Front')
    figname = 'pareto_{}_{}.png'.format(dim1, dim2)
    figpath = os.path.join('metric_analysis', figname)
    plt.savefig(figpath)
    plt.close()

metric_data = get_all_metrics()
metric_scores = metric_data.filter(regex='mean_test')
# del metric_scores['mean_test_f1']

pareto_indices = identify_pareto(metric_scores.to_numpy())
dims = permutations(list(metric_scores), 2) 
for dim in dims:
    plot_pareto_front(dim[0], dim[1], metric_data, pareto_indices, 1.5)

metric_data_pareto = metric_data.iloc[pareto_indices].reset_index()
metric_data_pareto = metric_data_pareto[['params', 'feature', 'model'] + list(metric_scores)]
metric_data_pareto.to_csv('metric_analysis/pareto_models.csv', index=False)



metric_data_pareto.describe(include=np.object)
print(metric_data_pareto['feature'].value_counts())
print(metric_data_pareto['model'].value_counts())
print((metric_data_pareto['feature'] + '+' + metric_data_pareto['model']).value_counts())


sorted_by = F1
sort_mapping = sort_mapping_master[sorted_by]
sortedby_dir = sort_mapping_master[sorted_by]['figdir']

metric_data_long, feature_of_interests= tranform_metrics_to_long(metric_data_pareto)

plt_generator = create_metric_barplot_features_one_metric(metric_data_long, sort_mapping, feature_of_interests)
save_figs(plt_generator, sortedby_dir)

plt_generator = create_metric_barplot_features_all_metric(metric_data_long, sort_mapping, feature_of_interests, topK=5)
save_figs(plt_generator, sortedby_dir)

plt_generator = create_metric_barplot_models_all_metric(metric_data_long, sort_mapping)
save_figs(plt_generator, sortedby_dir)