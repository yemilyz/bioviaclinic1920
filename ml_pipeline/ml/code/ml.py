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
import matplotlib.pyplot as plt


# numpy, pandas, and sklearn modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from sklearn.externals import joblib

# local ML modules
import datasets as datasets
import classifiers
# import preprocessors as preprocessors


######################################################################
# globals
######################################################################

# no magic numbers in code

N_ITER = 50    # number of parameter settings sampled (trade-off runtime vs quality)
CV = 10        # number of folds in cross-validation


######################################################################
# functions
######################################################################

def get_parser():
    """Make argument parser."""

    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument("dataset",
                        metavar="<dataset>",
                        choices=datasets.DATASETS,
                        help="[{}]".format(' | '.join(datasets.DATASETS)))
    parser.add_argument("classifier",
                        metavar="<classifier>",
                        choices=classifiers.CLASSIFIERS,
                        help="[{}]".format(' | '.join(classifiers.CLASSIFIERS)))

    # optional arguments
    # if preprocessors.PREPROCESSORS:
        # parser.add_argument("-p", "--preprocessor", dest="preprocessors",
        #                     metavar="<preprocessor>", 
        #                     default=[], action="append",
        #                     choices=preprocessors.PREPROCESSORS,
        #                     help="[{}]".format(' | '.join(preprocessors.PREPROCESSORS)))
    # else:
    parser.set_defaults(preprocessors=[])

    return parser



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
    # for pp in preprocessor_list:
    #     process = getattr(preprocessors, pp)()
    #     name = type(process).__name__
    #     transform = process.transformer_
    #     steps.append((name, transform))

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
    print()

    # precision, recall, f1
    p, r, f1, s = metrics.precision_recall_fscore_support(y_true, y_pred,
                                                          labels=labels,
                                                          average="weighted")
    print("precision: ",p)
    print("recall: ",r)
    print("f1: ", f1)
    # print report (redundant with above but easier)
    report = metrics.classification_report(y_true, y_pred, labels, target_names)
    print(report)

    return C, (a, p, r, f1)



def run(dataset, preprocessor_list, classifier):
    """Run ML pipeline.

    Parameters
    --------------------
        dataset           -- dataset, string
        preprocessor_list -- list of the 2 possible preprocessors
        classifier        -- classifier, string
    """

    # get dataset, then split into train and test set
    X, y, labels, target_names, feature_names = getattr(datasets, dataset)()
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
    n,d = X_train.shape

    # make pipeline
    pipe, param_grid = make_pipeline(preprocessor_list, classifier, n, d)

    # get param grid size
    sz = 1
    for vals in param_grid.values():
        sz *= len(vals)

    # tune model using randomized search
    n_iter = min(N_ITER, sz)    # cap max number of iterations
    search = RandomizedSearchCV(pipe, param_grid, n_iter=n_iter, cv=CV, refit='precision', scoring=['recall', 'precision', 'accuracy'])
    search.fit(X_train, y_train)
    print("Best parameters set found on development set:\n")
    print(search.best_params_)
    print("\n")

    # report results
    print("Detailed classification report (training set):\n")
    y_true, y_pred = y_train, search.predict(X_train)
    res_train = report_metrics(y_true, y_pred, labels, target_names)
    print("\n")

    print("Detailed classification report (test set):\n")
    y_true, y_pred = y_test, search.predict(X_test)
    res_test = report_metrics(y_true, y_pred, labels, target_names)
    print("\n")

    # fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.figure()
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.savefig("out.png")

    # save to file
    pp_string = ''
    for pp in preprocessor_list:
        if len(pp_string)>0:
            pp_string = pp_string+'_'
        pp_string = pp_string+pp

    prefix = os.path.join("results", '_'.join([dataset] + [pp_string] +[classifier]))

    # model
    joblib_file = prefix + "_pipeline.pkl"
    print(search.best_estimator_)
    joblib.dump(search.best_estimator_, joblib_file)

    pd.DataFrame(search.cv_results_).to_csv('data/'+classifier + 'cv_results.csv', index=False)

    # results
    json_file = prefix + "_results.json"
    res = {"C_train":      res_train[0].tolist(),
           "scores_train": res_train[1],
           "C_test":       res_test[0].tolist(),
           "scores_test":  res_test[1]}
    with open(json_file, 'w') as outfile:
        json.dump(res, outfile)
    return search


######################################################################
# main
######################################################################

def main():
    # set random seed (for repeatability)
    np.random.seed(42)

    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # main pipeline
    run(args.dataset, args.preprocessors, args.classifier)

if __name__ == "__main__":
    main()
    cv = pd.read_csv('data/RFcv_results.csv')
    print(list(cv))


# will fix plotting later
# ax1 = cv.plot.scatter('param_RF__max_depth', 'mean_train_score', color = 'darkorange', label ='train')
# cv.plot.scatter('param_RF__max_depth', 'mean_test_score', color = 'b', ax = ax1, label ='test')
# plt.title("CV for param_RF__max_depth")
# plt.savefig("results/cv_param_RF__max_depth.png")
# # plt.show()
# plt.close()


# ax1 = cv.plot.scatter('param_RF__max_features', 'mean_train_score', color = 'darkorange', label ='train')
# cv.plot.scatter('param_RF__max_features', 'mean_test_score', color = 'b', ax = ax1, label ='test')

# plt.title("CV for param_RF__max_features")
# plt.savefig("results/cv_param_RF__max_features.png")
# plt.close()


# ax1 = cv.plot.scatter('param_RF__n_estimators', 'mean_train_score', color = 'darkorange', label ='train')
# cv.plot.scatter('param_RF__n_estimators', 'mean_test_score', color = 'b', ax = ax1, label ='test')
# plt.title("CV for param_RF__n_estimators")
# plt.savefig("results/cv_param_RF__n_estimators.png")
# plt.close()
