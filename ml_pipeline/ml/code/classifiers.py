"""
Author      : Yi-Chieh Wu
Class       : HMC CS 121
Date        : 2018 Sep 20
Description : ML Classifiers
"""

# python modules
from abc import ABC

# numpy modules
import numpy as np

# sklearn modules
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


######################################################################
# classes
######################################################################

class Classifier(ABC):
    """Base class for classifier with hyper-parameter optimization.

    See sklearn.model_selection._search.

    Attributes
    --------------------
        estimator_  -- estimator object
            This is assumed to implement the scikit-learn estimator interface.
            Either estimator needs to provide a ``score`` function,
            or ``scoring`` must be passed.

        param_grid_ -- dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.

    Parameters
    --------------------
        n           -- int
            Number of samples.
        d           -- int
            Number of features.
    """

    def __init__(self, n, d):
        self.estimator_ = None
        self.param_grid_ = None


class Dummy(Classifier):
    """A Dummy classifier."""

    def __init__(self, n, d):
        self.estimator_ = DummyClassifier()
        self.param_grid_ = {}


class KNN(Classifier):
    """A kNN classifier."""

    def __init__(self, n, d):
        self.estimator_ = KNeighborsClassifier()
        self.param_grid_ = {"n_neighbors": np.arange(1,min(22,n),4)}


class RF(Classifier):
    """A Random Forest classifier."""

    def __init__(self, n, d):
        self.estimator_ = RandomForestClassifier()
        self.param_grid_ = {"n_estimators": np.arange(1,52,10),
                            "max_depth": np.arange(1,min(12,n),2),
                            "max_features": np.arange(1,min(12,d),2)}


class MLP(Classifier):
    """A Multi-Layer Perceptron classifier."""

    def __init__(self, n, d):
        self.estimator_ = MLPClassifier(max_iter=int(10e3))
        self.param_grid_ = {'hidden_layer_sizes': [(100,), (50,), (100, 100)]}


######################################################################
# globals
######################################################################

CLASSIFIERS = [c.__name__ for c in Classifier.__subclasses__()]
