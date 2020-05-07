import os 
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'savefig.format': 'svg'})
import seaborn as sns
sns.set()
# globals to help configure what metric to sort by
global F1, ROCAUC, sort_mapping_master
F1 = 'F1'
AUROC = 'AUROC'
F1_TRAIN = 'F1_TRAIN'
AUROC_TRAIN = 'AUROC_TRAIN'

sort_mapping_master = {
    F1: {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'metric':'f1'},
    AUROC: {'ranking':'rank_test_roc_auc', 'score':'mean_test_roc_auc', 'metric':'rocauc'},
    F1_TRAIN : {'ranking':'rank_test_f1', 'score':'mean_train_f1', 'metric':'f1_train'},
    AUROC_TRAIN : {'ranking':'rank_test_roc_auc', 'score':'mean_train_roc_auc', 'metric':'auroc_train'},
    }



def fixedWidthClusterMap(dataFrame, y_tick_fontsize, y_tick_rotation, row_colors, vmin):
    # clustermapParams = {
    #     'square':False # Tried to set this to True before. Don't: the dendograms do not scale well with it.
    # }
    figureWidth = 20
    figureHeight= 20

        # del feature_colors['feature']

    g = sns.clustermap(dataFrame,
        cmap='mako',
        vmin=vmin,
        vmax=1,
        figsize=(figureWidth,figureHeight),
        row_colors=row_colors[' '],
        # cbar_kws={'fontsize': 20},
        yticklabels=True)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = y_tick_fontsize, rotation = y_tick_rotation, va='bottom')
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(), fontsize = 20, rotation = 25, ha='right')
    g.ax_heatmap.set_xlabel("")
    g.ax_heatmap.set_ylabel("")

    cbar = g.ax_heatmap.collections[0].colorbar
    # here set the labelsize by 20
    cbar.ax.tick_params(labelsize=20)

    # Draw the legend bar for the classes 
    row_colors_unique = row_colors.drop_duplicates(subset='group', keep='first')
    row_colors_unique = row_colors_unique.sort_values(by=['group'])        
    for i, label in enumerate(row_colors_unique['group']):
        g.ax_col_dendrogram.bar(0, 0, color=row_colors_unique[' '][i],
                                label=label, linewidth=0)
    g.ax_col_dendrogram.legend(loc="upper right", ncol=3, fontsize=20, fancybox=False, framealpha=0)
    # Adjust the postion of the main colorbar for the heatmap
    # g.cax.set_position([.95, .2, .03, .45])
    return g

def get_ranked_metrics(sort_mapping, n_splits=10):
    ranking = sort_mapping['ranking']
    best_metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result_{}splits/'.format(n_splits), topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                if 'small' in metric_file or 'scrambled' in metric_file or 'uniform' in metric_file or 'python' in metric_file: continue
                # if not ('original_3' in metric_file or 'original_1' in metric_file): continue
                metric = pd.read_csv(metric_file)
                metric = metric.loc[metric[ranking]==1].head(1)
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                is_embedding = "embedding" in descriptor
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
    del metric_data_long['metric']
    return metric_data_long.sort_values(by=['score'], ascending=False)

def get_ranked_embedding_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==True].feature.unique()

def get_ranked_pc_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==False].feature.unique()

def get_feature_colors(metric_data_long):
    feature_grouping = metric_data_long['feature_grouping']
    feature_pal = sns.hls_palette(feature_grouping.unique().size, s=0.4)
    feature_lut = dict(zip(map(str, sorted(feature_grouping.unique())), feature_pal))
    feature_colors_series = pd.Series(feature_grouping).map(feature_lut)
    feature_colors = pd.DataFrame({' ': feature_colors_series.to_list(), 'feature': metric_data_long['feature'], 'group': feature_grouping})
    feature_colors.index = metric_data_long['feature']
    feature_colors = feature_colors.drop_duplicates(subset='feature', keep='first')
    del feature_colors['feature']
    return feature_colors



def main(): 
    model_rename = {
        'Dummy': 'Random',
        'GaussianBayes': "Gaussian Bayes",
        'LogiReg': 'Logistic Regression',
        'MLP': 'Multilayer Perceptron',
        'RF': 'Random Forest',
        'SVM': 'Support Vector Machine',
        'XGBoost': 'XGBoost',
    }

    for sorted_by, sort_mapping in sort_mapping_master.items():
        if sorted_by == AUROC_TRAIN or sorted_by == AUROC:
            vmin=0.5
        else:
            vmin=0

        metric_data = get_ranked_metrics(sort_mapping) # replace this line with pd.read_csv('metric_data_example.csv')
        metric_data_long = tranform_metrics_to_long(metric_data, sort_mapping['score'])

        metric_data_long = tranform_metrics_to_long(metric_data, sort_mapping['score'])
        metric_data_long['feature'] = metric_data_long['feature'].str.replace('feature_', '')
        metric_data_long['feature'] = metric_data_long['feature'].str.replace('embedding_', '')

        feature_grouping_all = []

        for feature, is_embedding in zip(metric_data_long['feature'], metric_data_long['is_embedding']):
            if is_embedding:
                # group = 'kmer{}_{}'.format(feature.split('_')[1], feature.split('_')[-1])
                group = 'kmer{}'.format(feature.split('_')[1])
            else:
                group = 'physicochemical'
            feature_grouping_all.append(group)
        
        metric_data_long['feature_grouping'] = feature_grouping_all    
        feature_colors = get_feature_colors(metric_data_long)

        del metric_data_long['feature_grouping']
        del metric_data_long['is_embedding']
        
        metric_data_wide = metric_data_long.pivot(index='feature', columns='model', values='score').sort_index()
        del metric_data_wide['KNN']
        metric_data_wide = metric_data_wide.rename(columns = model_rename)
        metric_data_wide.fillna(0, inplace = True)
        g = fixedWidthClusterMap(
            metric_data_wide,
            y_tick_fontsize=10,
            y_tick_rotation=45,
            row_colors=feature_colors,
            vmin=vmin,
        )
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle('Heatmap of Mean Validation {} Score'.format(sorted_by), fontsize=25)
        plt.savefig('metric_analysis/heatmap_{}_label.pdf'.format(sorted_by))
        plt.savefig('metric_analysis/heatmap_{}_label.png'.format(sorted_by))
        plt.close()


if __name__ == "__main__":
    main()

