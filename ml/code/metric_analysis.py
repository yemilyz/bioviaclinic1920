import os 
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
# globals to help configure what metric to sort by
global F1, ROCAUC, sort_mapping_master
F1 = 'F1'
ROCAUC = 'AUC_ROC'
sort_mapping_master = {
    F1: {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'metric':'f1'},
    ROCAUC: {'ranking':'rank_test_roc_auc', 'score':'mean_test_roc_auc', 'metric':'rocauc'},
    }



def fixedWidthClusterMap(dataFrame, y_tick_fontsize, y_tick_rotation):
    # clustermapParams = {
    #     'square':False # Tried to set this to True before. Don't: the dendograms do not scale well with it.
    # }
    figureWidth = 20
    figureHeight= 20
    g = sns.clustermap(metric_data_wide,
        cmap='vlag',
        vmin=vmin,
        vmax=1,
        figsize=(figureWidth,figureHeight),
        yticklabels=True)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = y_tick_fontsize, rotation = y_tick_rotation, va='bottom')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 20, rotation = 45, ha='right')
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")
    return g

def get_ranked_metrics(sort_mapping, n_splits=10):
    ranking = sort_mapping['ranking']
    best_metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result_{}splits/'.format(n_splits), topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                if 'small' in metric_file or 'scrambled' in metric_file or 'uniform' in metric_file: continue
                metric = pd.read_csv(metric_file)
                metric = metric.loc[metric[ranking]==1].head(1)
                print(metric.shape)
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                is_embedding = "embedding" in descriptor2
                metric['descriptor'] = descriptor
                metric['is_embedding'] = is_embedding
                if best_metrics_all.empty:
                    best_metrics_all = metric
                else:
                    best_metrics_all = best_metrics_all.append(metric)
    best_metrics_all = best_metrics_all.sort_values(by=['is_embedding'], ascending=True)
    return best_metrics_all

def tranform_metrics_to_long(metric_data, score):
    """ takes wide form metric data and returns long form wide data, as well as a sorted
    list of features (best to worst)
    """
    new_descriptor = metric_data["descriptor"].str.rsplit("_", n = 3, expand = True)
    is_embedding = metric_data["is_embedding"]
    metric_data = metric_data.filter(regex=score)
    value_list = list(metric_data)
    metric_data['feature'] = new_descriptor[0]
    model = new_descriptor[1]
    metric_data['model'] = model
    metric_data['is_embedding'] = is_embedding
    metric_data_long = pd.melt(
        metric_data,
        id_vars=['feature', 'model', 'is_embedding'],
        value_vars=value_list,
        value_name='score',
        var_name='metric'
        )
    # set_metric = metric_data_long['metric'].str.rsplit("_", n=1, expand=True)
    # new_metric = set_metric[1].str.replace('auc', 'rocauc')
    # new_set = set_metric[0].str.split("_", n=3, expand=True)[1]
    # metric_data_long['metric'] = new_metric
    del metric_data_long['metric']
    return metric_data_long.sort_values(by=['score'], ascending=False)

def get_ranked_embedding_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==True].feature.unique()

def get_ranked_pc_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==False].feature.unique()

def main():
    sorted_by = F1
    sort_mapping = sort_mapping_master[sorted_by]

    metric_data = get_ranked_metrics(sort_mapping) # replace this line with pd.read_csv('metric_data_example.csv')
    metric_data_long = tranform_metrics_to_long(metric_data, sort_mapping['score'])

    embedding_features = get_ranked_embedding_features(metric_data_long)
    pc_features = get_ranked_pc_features(metric_data_long)

    top_features = embedding_features[:3].tolist() + pc_features[:3].tolist()
    print(top_features)


# if __name__ == "__main__":
#     main()

for sorted_by, sort_mapping in sort_mapping_master.items():
    if sorted_by==ROCAUC:
        vmin=0.5
    else:
        vmin=0

    metric_data = get_ranked_metrics(sort_mapping) # replace this line with pd.read_csv('metric_data_example.csv')
    metric_data_long = tranform_metrics_to_long(metric_data, sort_mapping['score'])

    metric_data_long = tranform_metrics_to_long(metric_data, sort_mapping['score'])
    metric_data_long['feature'] = metric_data_long['feature'].str.replace('feature_', '')
    metric_data_long['feature'] = metric_data_long['feature'].str.replace('embedding_', '')

    for is_embedding, data in metric_data_long.groupby(['is_embedding']):
        metric_data_wide = data.pivot(index='feature', columns='model', values='score')
        if is_embedding:
            font = 10
        else:
            font = 20
        g = fixedWidthClusterMap(metric_data_wide, y_tick_fontsize=font, y_tick_rotation=45)
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle('Heatmap of Mean Validation {} Score'.format(sorted_by), fontsize=22)
        plt.savefig('metric_analysis/heatmap_{}_embedding={}.png'.format(sorted_by, is_embedding))
        plt.close()


    del metric_data_long['is_embedding']
    metric_data_wide = metric_data_long.pivot(index='feature', columns='model', values='score')

    g = fixedWidthClusterMap(metric_data_wide, y_tick_fontsize=10, y_tick_rotation=45)
    plt.subplots_adjust(top=0.95)
    g.fig.suptitle('Heatmap of Mean Validation {} Score'.format(sorted_by), fontsize=22)
    plt.savefig('metric_analysis/heatmap_{}.png'.format(sorted_by))
