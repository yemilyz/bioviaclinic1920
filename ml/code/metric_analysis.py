import os 
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18

sort_mapping_master = {
    'f1': {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'figdir': 'f1_sorted'},
    'roc_auc': {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'figdir': 'rocauc_sorted'},
    }
TOPK = 20

def get_ranked_metrics(sort_mapping):
    ranking = sort_mapping['ranking']
    score = sort_mapping['score']
    best_metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result/', topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                metric = pd.read_csv(metric_file)
                best_metric = metric.loc[metric[ranking]==1]
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                best_metric['descriptor'] = descriptor
                if best_metrics_all.empty:
                    best_metrics_all = best_metric
                else:
                    best_metrics_all = best_metrics_all.append(best_metric)
    best_metrics_all = best_metrics_all.sort_values(by=[score], ascending=False)
    return best_metrics_all

def create_metric_barplot_within(metric_data, sort_mapping, topK=TOPK):
    new_descriptor = metric_data["descriptor"].str.rsplit("_", n = 3, expand = True)
    metric_data = metric_data.filter(regex='split')
    value_list = list(metric_data)

    metric_data['feature'] = new_descriptor[0]
    metric_data['model'] = new_descriptor[1].str.replace('Dummy', '*Dummy')

    metric_data_long = pd.melt(
        metric_data,
        id_vars=['feature', 'model'],
        value_vars=value_list,
        value_name='score',
        var_name='metric'
        )
    new_metric = metric_data_long['metric'].str.rsplit("_", n = 1, expand = True)
    metric_data_long['metric'] = new_metric[1]


    for i, (feature_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('feature', sort=False)):
        print(feature_of_interest)
        metric_data_small = metric_data_small.sort_values(by=['model'])
        plt.figure(figsize=(15,12))
        ax = sns.barplot(
            x="metric",
            y="score",
            hue="model",
            palette="colorblind",
            ci='sd',
            errwidth=1,
            data=metric_data_small,
            )
        ax.set_title(feature_of_interest, fontsize=24)
        # plt.legend(loc='upper left')
        figtitle = '{}_model_metrics.png'.format(feature_of_interest)
        sortedby_dir = sort_mapping['figdir']
        plt.savefig(os.path.join('metric_analysis', sortedby_dir, figtitle))
        plt.close()
        if i == topK:
            break


sorted_by = 'f1'
metric_data = get_ranked_metrics(sort_mapping_master[sorted_by])
create_metric_barplot_within(metric_data, sort_mapping_master[sorted_by])