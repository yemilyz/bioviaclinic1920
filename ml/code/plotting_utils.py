import os

import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt

from plotting_params import params

plt.rcParams.update(params)
'''
input data schema looks like this:

              feature       model     metric     score    set  is_embedding
0  physico-chemical 1  GaussianNB       aupr  0.309366  train         False
1  physico-chemical 1  GaussianNB         f1  0.454762  train         False
2  physico-chemical 1  GaussianNB     rocauc  0.662075  train         False
3  physico-chemical 1  GaussianNB  precision  0.420705  train         False
'''


def create_metric_barplot_models_all_metric(metric_data_long):
    """
    returns generator of (figname, plt object) of metric bar plots given data and a dictonary for sorting.
    """
    for i, (feature_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('feature', sort=False)):
        # metric_data_small["model_set"] = metric_data_small["model"] + '_' +  metric_data_small["set"]
        metric_data_small.reindex(metric_data_small.mean().sort_values().index, axis=1)
        metric_data_small = metric_data_small.sort_values(by=['model', 'metric'], ascending=False)
        g = sns.catplot(
            x="metric",
            y="score",
            hue_order=[
                'Random',
                'Gaussian Bayes',
                'Logistic Regression',
                'Support Vector Machine',
                'Random Forest',
                'XGBoost',
                'Multilayer Perceptron',
            ],
            order=['au_pr', 'au_roc', 'f1', 'precision', 'recall'],
            hue="model",
            col="set",
            kind="bar",
            ci='sd',
            palette='muted',
            errwidth=0.3,
            errcolor='black',
            saturation=0.85,
            data=metric_data_small,
            )
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(feature_of_interest, x=0.42)
        g.set(ylim=(0, 1))
        figtitle = '{}_model_metrics'.format(feature_of_interest)
        yield (figtitle, plt)


def create_metric_barplot_features_all_metric(metric_data_long):
    """
    returns generator of (figname, plt object) of metric bar plots given data and a dictonary for sorting.
    """
    for _, (model_of_interest, metric_data_small) in enumerate(metric_data_long.groupby('model', sort=False)):
        # metric_data_small.reindex(metric_data_small.mean().sort_values().index, axis=1)
        # metric_data_small["feature_set"] = metric_data_small["feature"] + '_' +  metric_data_small["set"]
        metric_data_small = metric_data_small.sort_values(by=['metric', 'is_embedding'], ascending=False).reset_index()
        # plt.figure(figsize=(15,12))

        pal = sns.diverging_palette(220, 20, s=85, l=50, n=7)
        pal = pal[0], pal[1], pal[-1], pal[-2]
        g = sns.catplot(
            x="metric",
            y="score",
            order=['au_pr', 'au_roc', 'f1', 'precision', 'recall'],
            hue="feature",
            col="set",
            kind="bar",
            ci='sd',
            palette=pal,
            errwidth=0.3,
            errcolor='black',
            saturation=0.85,
            data=metric_data_small,
            )
        plt.subplots_adjust(top=0.85)
        g.fig.suptitle(model_of_interest, x=0.42)
        g.set(ylim=(0, 1))
        figtitle = '{}_model_all_metrics'.format(model_of_interest)
        yield (figtitle, plt)

def save_figs(plt_generator, sortedby_dir):
    """ saves the plots into desired directories
    """
    fig_directory = os.path.join('metric_analysis', sortedby_dir)
    if not os.path.exists(fig_directory):
        os.makedirs(fig_directory)
    for figname, metric_plot in plt_generator:
        metric_plot.savefig(os.path.join(fig_directory, '{}.pdf'.format(figname)))
        # plt.close()
        metric_plot.savefig(os.path.join(fig_directory, '{}.png'.format(figname)))
        plt.close()