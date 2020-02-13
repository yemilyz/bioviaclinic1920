import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from constant import DI_LABELS_CSV, FEATURE_DIR, FIGURE_DIR

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, valid_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    # fit_times_mean = np.zeros(train_scores_mean.shape)
    # fit_times_std = np.zeros(train_scores_mean.shape)

    # Plot learning curve
    axes.grid()
    print(train_sizes)
    print(train_scores_std)
    print(train_scores_mean)
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, valid_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # # Plot fit_time vs score
    # axes[2].grid()
    # axes[2].plot(fit_times_mean, valid_scores_mean, 'o-')
    # axes[2].fill_between(fit_times_mean, valid_scores_mean - valid_scores_std,
    #                      valid_scores_mean + valid_scores_std, alpha=0.1)
    # axes[2].set_xlabel("fit_times")
    # axes[2].set_ylabel("Score")
    # axes[2].set_title("Performance of the model")

    return plt, train_scores_mean, valid_scores_mean


fig, axes = plt.subplots(1, 2, figsize=(8, 16))

embed_features_dir0 = os.path.join(FEATURE_DIR, 'embedding_features', 'feature_embedding_original_5_7.csv')
win_features_dir = os.path.join(FEATURE_DIR, 'training_full.csv')
protparam_features_dir = os.path.join(FEATURE_DIR, 'protparam_features.csv')


X_p = pd.read_csv(protparam_features_dir, index_col=0)
X_p.index = X_p['name']
del X_p['name']

X_e0 = pd.read_csv(embed_features_dir0, index_col=0)
del X_e0['pdb_code']

y = pd.read_csv(DI_LABELS_CSV)
y.Name = y.Name.str.slice(stop=4)
X_p = X_p.loc[y.Name]
X_e0 = X_e0.loc[y.Name]

y['Developability Index (Fv)'] = (-1)*y['Developability Index (Fv)']
X_w = pd.read_csv(win_features_dir)
X_w.index = X_w['pdb_code']
del X_w['DI_all']
del X_w['Dev']
del X_w['pdb_code']
X_w = X_w.loc[y.Name]

X = X_e0
y = y['Developability Index (Fv)'] >= y['Developability Index (Fv)'].describe(percentiles=[0.6])[5]

title = "Learning Curves (Naive Bayes, Embedding_5_7)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = BernoulliNB()
plot, train, valid = plot_learning_curve(estimator, title, X, y, axes=axes[0], ylim=(0.4, 1.01),
                    cv=cv, n_jobs=4)

title = "Learning Curves (Random Forest, Embedding_5_7)"
# # SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=12345)
estimator = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=12345)

plot_learning_curve(estimator, title, X, y, axes=axes[1], ylim=(0.4, 1.01),
                    cv=cv, n_jobs=4)

plt.show()