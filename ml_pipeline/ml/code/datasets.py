"""
Author      : Yi-Chieh Wu
Class       : HMC CS 121
Date        : 2018 Sep 24
Description : ML Datasets
"""

# python modules
import sys
import inspect

# pandas module
import pandas as pd

######################################################################
# functions
######################################################################
def DI():
    """Load DI dataset"""

    target_names = ["Low", "High"]
    labels = [0, 1]

    # read csv
    df = pd.read_csv("data/labeled_data_modified_test.csv")

    # part b: process
    df = df.drop(['Name', 'DI Classification'], axis=1)
    X = df.drop("Developability Index (All)", axis=1)
    y = df["Developability Index (All)"]
    feature_names = X.columns

    return X, y, labels, target_names, feature_names


######################################################################
# globals
######################################################################

DATASETS = [name for name, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isfunction(obj)]
