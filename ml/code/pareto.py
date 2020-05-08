import os 
import glob
from itertools import permutations

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from plotting_params import params

plt.rcParams.update(params)


def get_all_metrics(n_splits=10):
    metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result_{}splits/'.format(n_splits), topdown=False):
        if 'AA' in root or 'KNN' in root:
            print(root, 'skipped')
            continue
        for name in dirs:     
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                if 'KNN' in metric_file:
                    print(metric_file, 'skipped')
                    continue
                metric = pd.read_csv(metric_file)
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                is_embedding = "embedding" in descriptor
                metric['descriptor'] = descriptor
                metric['is_embedding'] = is_embedding
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

def get_thresholded_data(metric_data, metric_thresholds):
    for metric, threshold in metric_thresholds.items():
        metric_data = metric_data.loc[metric_data[metric]>threshold]
    return metric_data.reset_index()


def plot_pareto_front(
    dim1,
    dim2,
    metric_data,
    metric_data_thresholded,
    pareto_indices,
    metric_thresholds,
    marker_size=1.5):
    metric_data_pareto = metric_data_thresholded.iloc[pareto_indices]

    x_all = metric_data[dim1]
    y_all = metric_data[dim2]

    x_thresh = metric_data_thresholded[dim1]
    y_thresh = metric_data_thresholded[dim2]

    x_pareto = metric_data_pareto[dim1]
    y_pareto = metric_data_pareto[dim2]

    x_line = metric_thresholds[dim1]
    y_line = metric_thresholds[dim2]
    
    plt.scatter(x_all, y_all, s=marker_size, alpha=0.2, color='gray')
    plt.scatter(x_thresh, y_thresh, s=marker_size, alpha=1)
    plt.scatter(x_pareto, y_pareto, s=marker_size, color='r')
    plt.axhline(y=y_line, xmin=x_line, xmax=1, linestyle='-.', linewidth=1, color='green')
    plt.vlines(x=x_line, ymin=y_line, ymax=1, linestyle='-.', linewidth=1, color='green')
    plt.xlabel(dim1)
    plt.ylabel(dim2)
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.grid(b=None)
    plt.title('Pareto Front')
    figname = 'pareto_{}_{}.png'.format(dim1, dim2)
    figpath = os.path.join('metric_analysis', figname)
    plt.savefig(figpath)
    plt.close()


def main():
    # globals to help configure what metric to sort by
    global F1, ROCAUC, RECALL, metric_thresholds
    F1 = 'mean_test_f1'
    ROCAUC = 'mean_test_roc_auc'
    RECALL = 'mean_test_recall'
    PRECISION = 'mean_test_precision'
    AUPR = 'mean_test_average_precision'
    metric_thresholds = {F1: 0.4, ROCAUC: 0.6, RECALL: 0.4, PRECISION: 0.4, AUPR: 0.6}


    metric_data = get_all_metrics()
    metric_data_thresholded = get_thresholded_data(metric_data, metric_thresholds)
    metric_scores = metric_data_thresholded.filter(regex='mean_test')
    pareto_indices = identify_pareto(metric_scores.to_numpy())

    dims = permutations(list(metric_scores), 2) 
    for dim in dims:
        plot_pareto_front(dim[0], dim[1], metric_data, metric_data_thresholded, pareto_indices, metric_thresholds, 1.5)

    metric_data_pareto = metric_data_thresholded.iloc[pareto_indices]
    metric_data_pareto = metric_data_pareto[['params', 'feature', 'model'] + list(metric_scores)]
    metric_data_pareto.to_csv('metric_analysis/pareto_models.csv', index=False)

    metric_data_pareto.describe(include=np.object)
    print(metric_data_pareto['feature'].value_counts())
    print(metric_data_pareto['model'].value_counts())
    print((metric_data_pareto['feature'] + '+' + metric_data_pareto['model']).value_counts())


if __name__ == "__main__":
    main()

