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
from constant import DI_LABELS_CSV

######################################################################
# functions
######################################################################
# def DI(feature_path):
#     """Load DI dataset"""

#     target_names = ["Low", "High"]
#     labels = [0, 1]

#     # read csv
#     DI_lables = pd.read_csv(DI_LABELS_CSV)
#     features = pd.read_csv(feature_path)

#     # part b: process
#     df = df.drop(['pdb_code', 'Developability Index (All)'], axis=1)
#     X = df.drop('DI Classification', axis=1)
#     y = df['DI Classification']
#     feature_names = X.columns

#     return X, y, labels, target_names, feature_names


def get_dataset(feature_path, label_path=DI_LABELS_CSV):
    y = pd.read_csv(label_path)
    y.Name = y.Name.str.slice(stop=4)
    X = pd.read_csv(feature_path, index_col=0)
    try:
        del X['name']
    except KeyError:
        try:
            del X['pdb_code']
        except KeyError:
            pass
    try:
        del X['Dev']
    except:
        pass
    try:
        del X['DI_all']
    except:
        pass
    X = X.loc[y.Name]
    feature_names = list(X)
    y = y['Developability Index (Fv)'] >= y['Developability Index (Fv)'].describe(percentiles=[0.75])[5]
    return X, y, feature_names


######################################################################
# globals
######################################################################

# DATASETS = [name for name, obj in inspect.getmembers(sys.modules[__name__]) if inspect.isfunction(obj)]
