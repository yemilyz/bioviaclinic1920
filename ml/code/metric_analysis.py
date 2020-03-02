import os 
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams.update({'figure.autolayout': True})

F1 = 'f1'
ROCAUC = 'roc_auc'

sort_mapping_master = {
    F1: {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'figdir': 'f1_sorted', 'metric':'f1'},
    ROCAUC: {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'figdir': 'rocauc_sorted', 'metric':'rocauc'},
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

def tranform_metrics_to_long(metric_data):
    new_descriptor = metric_data["descriptor"].str.rsplit("_", n = 3, expand = True)
    metric_data = metric_data.filter(regex='split')
    metric_data = metric_data.filter(regex='test')
    value_list = list(metric_data)

    metric_data['feature'] = new_descriptor[0]
    model = new_descriptor[1].str.replace('Dummy', '*Dummy')

    metric_data['model'] = model
    # metric_data['metric'] = metric

    metric_data_long = pd.melt(
        metric_data,
        id_vars=['feature', 'model'],
        value_vars=value_list,
        value_name='score',
        var_name='metric'
        )
    new_metric = metric_data_long['metric'].str.rsplit("_", n=1, expand=True)
    new_metric = new_metric[1].str.replace('auc', 'rocauc')
    metric_data_long['metric'] = new_metric
    feature_of_interests = metric_data_long.feature.unique()
    return metric_data_long, feature_of_interests



def create_metric_barplot_models_all_metric(metric_data_long, sort_mapping, topK=TOPK):
    for i, (feature_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('feature', sort=False)):
        metric_data_small = metric_data_small.sort_values(by=['model', 'metric'])
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
    

def create_metric_barplot_features_one_metric(metric_data_long, sort_mapping, feature_of_interests, topK=42):
    for i, (model_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('model', sort=False)):
        print(model_of_interest)
        # 
        metric_data_small = metric_data_small[metric_data_small['feature'].isin(feature_of_interests[:topK])]
        
        metric_data_small = metric_data_small.loc[metric_data_small['metric'] == sort_mapping['metric']]
        # metric_data_small = metric_data_small.sort_values(by=['score'], ascending=False).reset_index()

        sorted_result = metric_data_small.groupby('feature').agg(['mean'])
        sorted_result = sorted_result.sort_values(by=[('score','mean')], ascending=False)
        plt.figure(figsize=(15,12))
        ax = sns.barplot(
            x='feature',
            y="score",
            hue="metric",
            palette="colorblind",
            ci='sd',
            errwidth=1,
            data=metric_data_small,
            order=sorted_result.index,
            )
        ax.set_ylim(0,1)
        ax.set_title(model_of_interest, fontsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, horizontalalignment='right')
        plt.subplots_adjust(bottom=0.5)
        # plt.tight_layout()
        plt.legend(loc='upper right')
        figtitle = '{}_model_metrics.png'.format(model_of_interest)
        sortedby_dir = sort_mapping['figdir']
        plt.savefig(os.path.join('metric_analysis', sortedby_dir, figtitle))
        plt.close()


    
def create_metric_barplot_features_all_metric(metric_data_long, sort_mapping, feature_of_interests, topK=5):
    for i, (model_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('model', sort=False)):
        print(model_of_interest)
        # 
        metric_data_small = metric_data_small[metric_data_small['feature'].isin(feature_of_interests[:topK])]
        
        # metric_data_small = metric_data_small.loc[metric_data_small['metric'] == sort_mapping['metric']]
        metric_data_small = metric_data_small.sort_values(by=['metric'], ascending=False).reset_index()

   
        metric_data_small.reindex(metric_data_small.mean().sort_values().index, axis=1)
        plt.figure(figsize=(15,12))
        ax = sns.barplot(
            x='metric',
            y="score",
            hue="feature",
            palette="colorblind",
            ci='sd',
            errwidth=1,
            data=metric_data_small,
            # order=sorted_result.index,
            )
        ax.set_ylim(0,1)
        ax.set_title(model_of_interest, fontsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, horizontalalignment='right')
        plt.subplots_adjust(bottom=0.5)
        # plt.tight_layout()
        plt.legend(loc='upper right')
        figtitle = '{}_model_all_metrics.png'.format(model_of_interest)
        sortedby_dir = sort_mapping['figdir']
        plt.savefig(os.path.join('metric_analysis', sortedby_dir, figtitle))
        plt.close()



sorted_by = ROCAUC
metric_data = get_ranked_metrics(sort_mapping_master[sorted_by])
metric_data_long, feature_of_interests= tranform_metrics_to_long(metric_data)
create_metric_barplot_features_all_metric(metric_data_long, sort_mapping_master[sorted_by], feature_of_interests, topK=5)
create_metric_barplot_models_all_metric(metric_data_long, sort_mapping_master[sorted_by])
create_metric_barplot_features_one_metric(metric_data_long, sort_mapping_master[sorted_by], feature_of_interests)