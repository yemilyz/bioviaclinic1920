import os
import pickle
import glob

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.manifold import LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE

from constant import DI_LABELS_CSV, FEATURE_DIR, FIGURE_DIR

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


embed_features_dir0 = os.path.join(FEATURE_DIR, 'embedding_features', 'feature_embedding_original_5_7.csv')
win_features_dir = os.path.join(FEATURE_DIR, 'training_full.csv')
protparam_features_dir = os.path.join(FEATURE_DIR, 'protparam_features.csv')


sns.set_style('white')
sns.set_context('paper')

# Plot adjustments:
plt.rcParams.update({'ytick.labelsize': 14})
plt.rcParams.update({'xtick.labelsize': 14})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'legend.fontsize': 20})
plt.rcParams.update({'axes.titlesize': 14})
plt.rcParams.update({'axes.grid': False})
plt.rcParams.update({'legend.markerscale': 2})
plt.rcParams.update({'legend.fancybox': True})

# plt.rcParams['font.family'] = 'Oswald'
def plot_PCA(X, y, title, n_components, pc1=1, pc2=2):
    # Standardizing the features
    # X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['pc{}'.format(i) for i in range(1, n_components+1)])
    finalDf = pd.concat([principalDf, y], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component {}'.format(pc1), fontsize = 15)
    ax.set_ylabel('Principal Component {}'.format(pc2), fontsize = 15)
    ax.set_title('{} component PCA'.format(n_components), fontsize = 20)
    targets = [True, False]
    colors = ['#16ff8b','#4c4cff']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['binary_labs'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc{}'.format(pc1)]
                , finalDf.loc[indicesToKeep, 'pc{}'.format(pc2)]
                , c = color
                , s = 50
                , alpha=0.7)
    ax.legend(targets)
    plt.title(title)
    ax.grid()


def plot_embed_pca(embed_feature_path, y, n_components=2):
    X = pd.read_csv(embed_feature_path, index_col=0)
    del X['pdb_code']
    X = X.loc[y.Name]
    title = embed_feature_path.split('/')[-1].split('.')[0]
    plot_PCA(X, y[['binary_labs']], title, n_components, pc1=1, pc2=2)
    figname = title + '_pca.png'
    figpath = os.path.join(FIGURE_DIR, 'embedding', figname)
    plt.savefig(figpath)
    plt.close()



embedding_dir = os.path.join(FEATURE_DIR, 'embedding_features/*.csv')
y = pd.read_csv(DI_LABELS_CSV)
y.Name = y.Name.str.slice(stop=4)
y['Developability Index (Fv)'] = (-1)*y['Developability Index (Fv)']
y['binary_labs'] = y['Developability Index (Fv)'] >= y['Developability Index (Fv)'].describe(percentiles=[0.7])[5]

for embed_feature_path in glob.glob(embedding_dir):
    print(embed_feature_path)
    plot_embed_pca(embed_feature_path, y)



def plot_projection(X, titles, df, n_parents, cmap='winter', col='binary_labs', **plot_args):
    fig = plt.figure(figsize=(20, 4.7))
    gs = gridspec.GridSpec(1, len(X), width_ratios=[6] * len(X), height_ratios=[6])
    axs = [plt.subplot(ggs) for ggs in gs]
    # _ = fig.tight_layout()
    pal = sns.color_palette('deep')
    # return
    for i, (x, ax) in enumerate(zip(X, axs)):
        print(i)
        mani = TSNE(perplexity=50, random_state=0, n_iter=2000, learning_rate=50.0)
        print(x.shape)
        x = StandardScaler().fit_transform(x)
        low_dim_embs = mani.fit_transform(x)
        print(low_dim_embs.shape)
        handles = []         
        sc = ax.scatter(low_dim_embs[:, 0], low_dim_embs[:, 1], c=df[col], cmap=cmap, **plot_args)
        if n_parents > 0:
            handles.append(ax.plot(low_dim_embs[:n_parents, 0], low_dim_embs[:n_parents, 1], 
                                    '^', markersize=25, color=pal[2], label='parent')[0])
        # ax.tick_params(
        #     axis='both',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     # bottom='off',      # ticks along the bottom edge are off
        #     # top='off',  
        #     # left='off',       # ticks along the top edge are off
        #     # right='off',
        #     labelbottom='off',
        #     labeltop='off',
        #     labelleft='off',
        #     labelright='off',
        # ) # labels along the bottom edge are off 
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.xaxis.set_major_formatter(plt.NullFormatter())

        # ax.tick_params(
        #     axis='y',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     left='off',      # ticks along the bottom edge are off
        #     right='off',         # ticks along the top edge are off
        #     labelleft='off') # labels along the bottom edge are off
        # Force them to be square
        if i == len(X)-1:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles, labels = sc.legend_elements()
            print('labels', labels)
            ax.legend(
                handles=handles,
                labels=['Low', 'High'],
                title="Developability", fontsize=12,
                title_fontsize = 14,
                bbox_to_anchor=(1.01, 0.5),
                loc="lower left"
                )

        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.title.set_text(titles[i])

    # cb = fig.colorbar(sc, cax=axs[-1], boundaries=[-1, 0, 1])
    # cb.ax.yaxis.set_label_position('left')
    # cb.set_label('DI')
    fig.suptitle("t-SNE Results", fontsize=20)
    # fig.tight_layout()
    plt.subplots_adjust(top=0.85)
    return fig, axs



def plot_ChRs():
    X_e0 = pd.read_csv(embed_features_dir0, index_col=0)
    del X_e0['pdb_code']
    # print(list(X))

    # X_e1 = pd.read_csv(embed_features_dir1, index_col=0)
    # del X_e1['name']

    X_w = pd.read_csv(win_features_dir)
    X_w.index = X_w['pdb_code']
    del X_w['DI_all']
    del X_w['Dev']
    del X_w['pdb_code']


        
    X_p = pd.read_csv(protparam_features_dir, index_col=0)
    X_p.index = X_p['name']
    del X_p['name']


    y = pd.read_csv(DI_LABELS_CSV)
    y.Name = y.Name.str.slice(stop=4)
    titles = ['protparam', 'embed']
    # print(y['Developability Index (Fv)'].describe(percentiles=0.2))
    y['Developability Index (Fv)'] = (-1)*y['Developability Index (Fv)']
    y['binary_labs'] = y['Developability Index (Fv)'] >= y['Developability Index (Fv)'].describe(percentiles=[0.6])[5]
    X_e0 = X_e0.loc[y.Name]
    # X_e1 = X_e1.loc[y.Name]
    X_w = X_w.loc[y.Name]
    X_p = X_p.loc[y.Name]
    # print(X.head())
    # print(y.head())
    plot_args = {
        's': 24,
        'alpha': 0.7,
    }
    return plot_projection([X_p, X_e0], titles, y, 0, **plot_args)

# plot_ChRs()
# plt.show()
# plt.savefig(DATA_DIR)

# y = pd.read_csv(DI_LABELS_CSV)
# y.Name = y.Name.str.slice(stop=4)
# print(y['Developability Index (Fv)'].describe(percentiles=0.2))
# y['binary_labs'] = y['Developability Index (Fv)'] < y['Developability Index (Fv)'].describe(percentiles=[0.2])[4] 
# plt.hist([y['Developability Index (Fv)'][y['Developability Index (Fv)'] < y['Developability Index (Fv)'].describe(percentiles=[0.2])[4]],
#     y['Developability Index (Fv)'][y['Developability Index (Fv)'] >= y['Developability Index (Fv)'].describe(percentiles=[0.2])[4]],
#     ], color = ['#16ff8b', '#4c4cff'])
# plt.title('DI Lables')
# plt.show()


# fig, ax = plt.subplots()

# y['Developability Index (Fv)'] = (-1)*y['Developability Index (Fv)']
# N, bins, patches = ax.hist(y['Developability Index (Fv)'], edgecolor='white', linewidth=1, bins=20)
# twenty = sum(bins<y['Developability Index (Fv)'].describe(percentiles=[0.8])[5]) - 1

# for i in range(0,twenty):
#     patches[i].set_facecolor('#4c4cff')
# for i in range(twenty, len(patches)):
#     patches[i].set_facecolor('#16ff8b')
# plt.xlabel('Developability Index', fontname="Oswald")
# fig.tight_layout()
# plt.show()

