import os 
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# seaborn and matplotlib defaults
sns.set(style="whitegrid")
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams.update({'figure.autolayout': True})


# globals to help configure what metric to sort by
global F1, ROCAUC, sort_mapping_master
F1 = 'f1'
ROCAUC = 'roc_auc'
sort_mapping_master = {
    F1: {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'figdir': 'f1_sorted', 'metric':'f1'},
    ROCAUC: {'ranking':'rank_test_roc_auc', 'score':'mean_test_roc_auc', 'figdir': 'rocauc_sorted', 'metric':'rocauc'},
    }


def get_ranked_metrics(sort_mapping, n_splits=10):
    ranking = sort_mapping['ranking']
    score = sort_mapping['score']
    best_metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result_{}splits/'.format(n_splits), topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                metric = pd.read_csv(metric_file)
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                is_embedding = "embedding" in descriptor
                metric['descriptor'] = descriptor
                metric['is_embedding'] = is_embedding
                if best_metrics_all.empty:
                    best_metrics_all = metric
                else:
                    best_metrics_all = best_metrics_all.append(metric)
    best_metrics_all = best_metrics_all.sort_values(by=['is_embedding', score], ascending=False)
    return best_metrics_all

def tranform_metrics_to_long(metric_data):
    """ takes wide form metric data and returns long form wide data, as well as a sorted
    list of features (best to worst)
    """
    new_descriptor = metric_data["descriptor"].str.rsplit("_", n = 3, expand = True)
    is_embedding = metric_data["is_embedding"]
    metric_data = metric_data.filter(regex='split')
    value_list = list(metric_data)
    # print(value_list)

    metric_data['feature'] = new_descriptor[0]
    model = new_descriptor[1].str.replace('Dummy', '*Dummy')
    metric_data['model'] = model
    metric_data['is_embedding'] = is_embedding
    metric_data_long = pd.melt(
        metric_data,
        id_vars=['feature', 'model', 'is_embedding'],
        value_vars=value_list,
        value_name='score',
        var_name='metric'
        )
    set_metric = metric_data_long['metric'].str.rsplit("_", n=1, expand=True)
    new_metric = set_metric[1].str.replace('auc', 'rocauc')
    new_set = set_metric[0].str.split("_", n=3, expand=True)[1]
    metric_data_long['metric'] = new_metric
    metric_data_long['set'] = new_set
    return metric_data_long

def get_ranked_embedding_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==True].feature.unique()

def get_ranked_pc_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==False].feature.unique()

def create_metric_barplot_features_one_metric(metric_data_long, sort_mapping, feature_of_interests, topK=15):
    """
    topK: the number of features to show in one figure
    returns generator of (figname, plt object) of metric bar plots given data and a dictonary for sorting
    """
    for i, (model_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('model', sort=False)):
        metric_data_small = metric_data_small[metric_data_small['feature'].isin(feature_of_interests[:topK])]
        metric_data_small = metric_data_small.loc[metric_data_small['metric'] == sort_mapping['metric']]
        metric_data_small = metric_data_small.sort_values(by=['set'], ascending=False)

        sorted_result = metric_data_small.loc[metric_data_small['set']=='test'].groupby(['feature']).agg(['mean'])
        sorted_result = sorted_result.sort_values(by=[('score','mean')], ascending=False)

        f, ax = plt.subplots(figsize=(15,12))
        sns.barplot(
            x='feature',
            y="score",
            hue="set",
            palette="colorblind",
            ci='sd',
            errwidth=2.5,
            data=metric_data_small.loc[metric_data_small['set']=='test'],
            order=sorted_result.index,
            )
        sns.barplot(
            x='feature',
            y="score",
            hue="set",
            palette="colorblind",
            ci='sd',
            errwidth=1,
            alpha=0.4,
            data=metric_data_small.loc[metric_data_small['set']=='train'],
            order=sorted_result.index,
            )
        sns.despine(left=True, bottom=True)
        ax.set_ylim(0,1)
        ax.set_title('{} | {} score'.format(model_of_interest, sort_mapping['metric']), fontsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, horizontalalignment='right')
        plt.subplots_adjust(bottom=0.5)
        plt.legend(loc='upper right')
        figtitle = '{}_model_metrics.png'.format(model_of_interest)
        yield (figtitle, plt)
    
def create_metric_barplot_features_all_metric(metric_data_long, sort_mapping, feature_of_interests, topK=5):
    """
    topK: the number of features to show in one figure
    returns generator of (figname, plt object) of metric bar plots given data and a dictonary for sorting.
    """
    for i, (model_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('model', sort=False)):
        metric_data_small = metric_data_small[metric_data_small['feature'].isin(feature_of_interests[:topK])]
        metric_data_small.reindex(metric_data_small.mean().sort_values().index, axis=1)
        metric_data_small["feature_set"] = metric_data_small["feature"] + '_' +  metric_data_small["set"]
        metric_data_small = metric_data_small.sort_values(by=['metric', 'is_embedding'], ascending=False).reset_index()
        
        plt.figure(figsize=(15,12))
        f, ax = plt.subplots(figsize=(15,12))
        sns.barplot(
            x='metric',
            y="score",
            hue="feature",
            palette="colorblind",
            ci='sd',
            errwidth=2.5,
            data=metric_data_small.loc[metric_data_small['set']=='test'],
            )
        handles, labels = ax.get_legend_handles_labels()
        labels = ['embedding 1', 'embedding 2', 'physico-chemical 1', 'physico-chemical 2']
        sns.barplot(
            x='metric',
            y="score",
            hue="feature",
            palette="colorblind",
            ci='sd',
            errwidth=1,
            data=metric_data_small.loc[metric_data_small['set']=='train'],
            alpha=0.4
            )
        sns.despine(left=True, bottom=True)
        ax.set_ylim(0,1)
        ax.set_title(model_of_interest, fontsize=24)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=14, rotation=45, horizontalalignment='right')
        plt.subplots_adjust(bottom=0.5)
        plt.legend(handles, labels, loc='upper left')
        figtitle = '{}_model_all_metrics.png'.format(model_of_interest)
        yield (figtitle, plt)


def create_metric_barplot_models_all_metric(metric_data_long, sort_mapping, topK=15):
    """
    topK: the number of features plot (equals the number of plots returned)
    returns generator of (figname, plt object) of metric bar plots given data and a dictonary for sorting.
    """
    for i, (feature_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('feature', sort=False)):
        metric_data_small["model_set"] = metric_data_small["model"] + '_' +  metric_data_small["set"]
        metric_data_small = metric_data_small.sort_values(by=['model_set', 'metric'], ascending=False)
        f, ax = plt.subplots(figsize=(15,12))
        sns.barplot(
            x="metric",
            y="score",
            hue="model",
            palette="colorblind",
            ci='sd',
            errwidth=2.5,
            data=metric_data_small.loc[metric_data_small['set']=='test'],
            )
        handles, labels = ax.get_legend_handles_labels()

        sns.barplot(
            x="metric",
            y="score",
            hue="model",
            palette="colorblind",
            ci='sd',
            errwidth=1,
            data=metric_data_small.loc[metric_data_small['set']=='train'],
            alpha=0.4, 
            )
        sns.despine(left=True, bottom=True)
        ax.set_title(feature_of_interest, fontsize=24)
        plt.legend(handles, labels, loc='upper right')
        figtitle = '{}_model_metrics.png'.format(feature_of_interest)
        yield (figtitle, plt)
        if i == topK:
            break

def save_figs(plt_generator, sortedby_dir):
    """ saves the plots into desired directories
    """
    fig_directory = os.path.join('metric_analysis', sortedby_dir)
    if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)
    for figname, metric_plot in plt_generator:
        metric_plot.savefig(os.path.join(fig_directory, figname))
        plt.close()


def main():
    sorted_by = F1
    sort_mapping = sort_mapping_master[sorted_by]
    sortedby_dir = sort_mapping_master[sorted_by]['figdir']

    metric_data = get_ranked_metrics(sort_mapping) # replace this line with pd.read_csv('metric_data_example.csv')
    metric_data_long = tranform_metrics_to_long(metric_data)

    embedding_features = get_ranked_embedding_features(metric_data_long)
    pc_features = get_ranked_pc_features(metric_data_long)

    top_features = embedding_features[:2].tolist() + pc_features[:2].tolist()
    plt_generator = create_metric_barplot_features_all_metric(metric_data_long, sort_mapping, top_features, topK=4)
    save_figs(plt_generator, sortedby_dir)


if __name__ == "__main__":
    main()

