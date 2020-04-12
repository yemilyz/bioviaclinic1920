"""
Author      : Yi-Chieh Wu
Class       : HMC CS 121
Date        : 2018 Sep 25
Description : Main ML Pipeline
"""

# python modules
import os
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import glob
import warnings
warnings.filterwarnings("ignore")
# warnings.simplefilter(action='ignore', category=FutureWarning)


# numpy, pandas, and sklearn modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, \
    LeaveOneOut, LeavePOut, StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn import metrics
from sklearn.externals import joblib

# local ML modules
from datasets import get_dataset
import classifiers
from constant import DI_LABELS_CSV, PROTPARAM_FEATURES, SLIDING_WIN_FEATURES, EMBEDDING_5_7_FEATURES
from learning_curve import plot_learning_curve
import preprocessors as preprocessors




######################################################################
# globals
######################################################################

# no magic numbers in code

N_ITER = 100    # number of parameter settings sampled (trade-off runtime vs quality)
CV_train = StratifiedKFold(n_splits=10, random_state=0)       # number of folds in cross-validation
CV_lc = StratifiedKFold(n_splits=10, random_state=0)

######################################################################
# functions
######################################################################

# def get_parser():
#     """Make argument parser."""

#     parser = argparse.ArgumentParser()

#     # positional arguments
#     parser.add_argument("dataset",
#                         metavar="<dataset>",
#                         choices=datasets.DATASETS,
#                         help="[{}]".format(' | '.join(datasets.DATASETS)))
#     parser.add_argument("classifier",
#                         metavar="<classifier>",
#                         choices=classifiers.CLASSIFIERS,
#                         help="[{}]".format(' | '.join(classifiers.CLASSIFIERS)))

#     # optional arguments
#     # if preprocessors.PREPROCESSORS:
#         # parser.add_argument("-p", "--preprocessor", dest="preprocessors",
#         #                     metavar="<preprocessor>", 
#         #                     default=[], action="append",
#         #                     choices=preprocessors.PREPROCESSORS,
#         #                     help="[{}]".format(' | '.join(preprocessors.PREPROCESSORS)))
#     # else:
#     parser.set_defaults(preprocessors=[])

#     return parser



def make_pipeline(preprocessor_list, classifier, n, d):
    """Make ML pipeline.

    Parameters
    --------------------
        preprocessor_list -- list of the 2 possible preprocessors
        classifier        -- classifier, string
        n                 -- number of samples, int
        d                 -- number of features, int
    """
    steps = []
    param_grid = {}

    # get preprocessor(s) and hyperparameters to tune using cross-validation
    for pp in preprocessor_list:
        process = getattr(preprocessors, pp)()
        name = type(process).__name__
        transform = process.transformer_
        steps.append((name, transform))

    # get classifier and hyperparameters to tune using cross-validation
    clf = getattr(classifiers, classifier)(n,d)
    name = type(clf).__name__
    transform = clf.estimator_
    steps.append((name, transform))
    for key, val in clf.param_grid_.items():
        param_grid[name + "__" + key] = val

    # stitch together preprocessors and classifier
    pipe = Pipeline(steps)
    return pipe, param_grid


def report_metrics(y_true, y_pred, labels=None, target_names=None):
    """Report main classification metrics.

    Parameters
    --------------------
        y_true       -- ground truth (correct) target values, array of shape (n_samples,)
        y_pred       -- estimated target values returned by classifier, array of shape (n_samples,)
        labels       -- list of label indices to include in report, list of strings
        target_names -- display names matching labels (same order), list of strings

    Return
    --------------------
        C      -- confusion matrix, see sklearn.metrics.confusion_matrix
        a      -- accuracy score, see sklearn.metrics.accuracy_score
        p      -- precision score, see sklearn.metrics.precision_score
        r      -- recall score, see sklearn.metrics.recall_score
        f1     -- f1 score, see sklearn.metrics.f1_score
    """

    # confusion matrix, then wrap in pandas to pretty print
    C = metrics.confusion_matrix(y_true, y_pred, labels)
    df = pd.DataFrame(C, columns=target_names, index=target_names)
    print("Confusion Matrix\n", df)
    print()

    # accuracy
    a = metrics.accuracy_score(y_true, y_pred)
    print("accuracy: ", a)
    # print()

    # precision, recall, f1
    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          labels=labels,
                                                          average="weighted")
    print("precision: ",p)
    print("recall: ",r)
    print("f1: ", f1)
    # print report (redundant with above but easier)
    # report = metrics.classification_report(y_true, y_pred, labels, target_names)
    # print(report)

    return df


def reportCV(cv_data):
    cv_data.fillna(0, inplace=True)
    for colname in cv_data.columns.tolist():
        if colname.startswith('param_'):
            fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)
            cv_data.plot.scatter(colname, 'mean_test_accuracy', color = 'darkorange', s=8, ax=axes[0], label ='accuracy')
            cv_data.plot.scatter(colname, 'mean_test_precision', color = 'b', s=8, ax=axes[1], label ='precision')
            cv_data.plot.scatter(colname, 'mean_test_precision', color = 'darkgreen', s=8, ax=axes[2], label ='recall')
            axes[1].set_title("CV test metrics for {}".format(colname),fontsize= 12) # title of plot
            # cv_data.plot.scatter(colname, 'mean_test_precision', color = 'b', s=3, alpha = 0.4, label ='precision')
            plt.savefig("results/cv_{}.png".format(colname))
            plt.close()





def run_one_featureset(
    feature_path,
    preprocessor_list,
    classifier,
    scoring,
    label_path=DI_LABELS_CSV,
    labels = [0, 1],
    target_names = ['Low', 'High'],
    iterations=N_ITER,
    n_jobs=4,
    n_splits=10,
    ):
    """Run ML pipeline.

    Parameters
    --------------------
        dataset           -- dataset, string
        preprocessor_list -- list of the 2 possible preprocessors
        classifier        -- classifier, string
    """

    # get dataset, then split into train and test set
    X, y, feature_names = get_dataset(feature_path, label_path)
    # print('\n'.join(feature_names))
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    y_train = y_train >= y_train.describe(percentiles=[0.8 ])[5]
    n,d = X_train.shape

    # make pipeline
    pipe, param_grid = make_pipeline(preprocessor_list, classifier, n, d)

    # get param grid size
    sz = 1
    try:
        for vals in param_grid.values():
            sz *= len(vals)
            # tune model using randomized search
        n_iter = min(iterations, sz)    # cap max number of iterations
    except TypeError:
        n_iter = iterations

    search = RandomizedSearchCV(
        pipe,
        param_grid,
        verbose=2,
        n_iter=n_iter,  
        cv=CV_train,
        refit=scoring,
        scoring=['recall', 'precision', 'average_precision', 'f1', 'roc_auc'],
        return_train_score=True,
        n_jobs=n_jobs)
    
    search.fit(X_train, y_train)
    print("Best parameters set found on development set:\n")
    print(search.best_params_)
    print("\n")

    # report results
    print("Detailed classification report (training set):\n")
    y_true, y_pred = y_train, search.predict(X_train)
    conf_mat = report_metrics(y_true, y_pred, labels, target_names)
    
    print("\n")

    # print("Detailed classification report (test set):\n")
    # y_true, y_pred = y_test, search.predict(X_test)
    # res_test = report_metrics(y_true, y_pred, labels, target_names)
    # print("\n")


    
    # reportCV(cv_data)

    dataset_name = os.path.split(feature_path)[-1].split('.')[0]

    results_dir = os.path.join('result_{}splits'.format(n_splits), dataset_name)
    prefix_metric = os.path.join(results_dir, 'metric')
    prefix_figure = os.path.join(results_dir, 'figure')
    prefix_model = os.path.join(results_dir, 'model')

    

    for directory in [prefix_metric, prefix_figure, prefix_model]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
    y_score = search.predict_proba(X_train)[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title('{} ROC Training'.format(classifier))
    axes[0].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    axes[0].legend(loc = 'lower right')
    axes[0].plot([0, 1], [0, 1],'r--')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_xlabel('False Positive Rate')
    

    # save to file
    pp_string = ''
    for pp in preprocessor_list:
        if len(pp_string)>0:
            pp_string = pp_string+'_'
        pp_string = pp_string+pp

    discriptor = "{}_{}".format(dataset_name, classifier)

    conf_mat.to_csv(os.path.join(prefix_metric, discriptor + "_confusion_matrix.csv"))

    # plt.savefig(os.path.join(prefix_figure, discriptor + '_roc_train.png'))
    # plt.close()

    lc_title = "{} Learning Curves".format(classifier)
    plot, train_scores_mean, train_scores_std, valid_scores_mean, valid_scores_std = plot_learning_curve(
        estimator=search.best_estimator_,
        title=lc_title,
        X=X_train,
        y=y_train,
        axes=axes[1],
        ylim=(0, 1.01),
        cv=CV_lc,
        n_jobs=n_jobs,
        train_sizes=np.linspace(.1, 1.0, 5),
        scoring=scoring,
        )
    plot.savefig(os.path.join(prefix_figure, discriptor + '_roc_lc.png'))
    plot.close()

    learning_curve_file = os.path.join(prefix_metric, discriptor + "_learning_curve.csv")
    
    learning_curve_data = {
        'train_scores_mean': train_scores_mean,
        'train_scores_std': train_scores_std,
        'valid_scores_mean': valid_scores_mean,
        'valid_scores_std': valid_scores_std,
        }
    pd.DataFrame.from_dict(learning_curve_data).to_csv(learning_curve_file)
    # model
    joblib_file = os.path.join(prefix_model, discriptor + "_pipeline.pkl")
    joblib.dump(search.best_estimator_, joblib_file)

    # results
    # json_file = os.path.join(prefix_metric, discriptor + "_results.json")
    # res = {"C_train":      res_train[0].tolist(),
    #        "scores_train": res_train[1],
    #     }
        #    "C_test":       res_test[0].tolist(),
        #    "scores_test":  res_test[1]}
    # with open(json_file, 'w') as outfile:
    #     json.dump(res, outfile)

    cv_data = pd.DataFrame(search.cv_results_)
    # print(cv_data)
    cv_data.to_csv(os.path.join(prefix_metric, discriptor + '_cv_metrics.csv'))

    roc_file = os.path.join(prefix_metric, discriptor  + "_roc.csv")
    with open(roc_file, 'w') as outfile:
        wr = csv.writer(outfile)
        wr.writerow(fpr)
        wr.writerow(tpr)
        wr.writerow(threshold)

    # return search


######################################################################
# main
######################################################################

def main():
    # set random seed (for repeatability)
    np.random.seed(42)
    # # parse arguments
    # parser = get_parser()
    # args = parser.parse_args()

    # main pipeline
    feature_paths = glob.glob('data/feature_*')
    embed_feature_paths = glob.glob('data/embedding_features/feature_*')

    for feature_path in feature_paths + embed_feature_paths:
        print('training for feature', feature_path)
        if 'msa' not in feature_path:
            continue
        for clf in classifiers.CLASSIFIERS:
            print('training ', clf)
            if clf == 'MLP' or clf == 'SVM' or clf == 'RF':
                iterations = 50
            elif clf == 'XGBoost':
                iterations = 20
            else:
                iterations = N_ITER
            n_splits = 10
            run_one_featureset(
                feature_path=feature_path,
                preprocessor_list=preprocessors.PREPROCESSORS,
                classifier=clf,
                scoring='f1',
                label_path=DI_LABELS_CSV,
                labels = [0, 1],
                target_names = ['Low', 'High'],
                iterations = iterations,
                n_jobs=-1,
                n_splits=n_splits,
                )

if __name__ == "__main__":
    main()
