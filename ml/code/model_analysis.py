import pickle
from joblib import dump, load
import glob
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn import metrics


from datasets import get_dataset
import preprocessors as preprocessors
from plotting_utils import create_metric_barplot_models_all_metric, \
    create_metric_barplot_features_all_metric, save_figs



np.random.seed(1234)

def calculate_metrics(y_true, y_pred):
    aupr = metrics.average_precision_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    rocauc = metrics.roc_auc_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    return (aupr, f1, rocauc, precision, recall)

def update_metric_data(metric_data, new_metrics, feature, model, subset):
    metric_data = metric_data.append(pd.DataFrame.from_dict(
        {'feature': [feature] * 5,
        'model': [model] * 5, 
        'metric': ['aupr', 'f1', 'auroc', 'precision', 'recall'],
        'score': new_metrics, 
        'set': [subset] * 5
        }),
        ignore_index=True,
    )
    return metric_data

def main():
    feature_namemap = {
            'feature_msa_avg': 'physicochemical 1',
            'feature_protparam':'physicochemical 2',
            'feature_embedding_original_3_7': 'embedding 1',
            'feature_embedding_original_3_5': 'embedding 2',
        }

    n_iter_bootstrap = 10
    models_of_interests = {
        'Random',
        # 'GaussianNB',
        'Logistic Regression',
        'Random Forest',
        'Multilayer Perceptron',
        'Support Vector Machine',
        }

    model_rename = {
        'DummyClassifier': 'Random',
        'GaussianNB': "Gaussian Bayes",
        'LogisticRegression': 'Logistic Regression',
        'MLPClassifier': 'Multilayer Perceptron',
        'RandomForestClassifier': 'Random Forest',
        'SVC': 'Support Vector Machine',
        'GradientBoostingClassifier': 'XGBoost',
    }


    metric_data = pd.DataFrame(columns=['feature', 'model', 'metric', 'score', 'set'])
    for feature_old, feature_new in feature_namemap.items():
        model_path_regex = 'result_10splits/{}/model/*.pkl'.format(feature_old)
        if 'embedding' in feature_old:
            feature_path = 'data/embedding_features/{}.csv'.format(feature_old)
        else:
            feature_path = 'data/{}.csv'.format(feature_old)
        model_paths = glob.glob(model_path_regex)
        for model_path in model_paths:
            pipe = load(model_path)
            model = model_rename[type(pipe[-1]).__name__]
            # if model not in models_of_interests: continue
            X, y, feature_names = get_dataset(feature_path)
            X_train, X_test, y_train_numeric, y_test_numeric = \
                    train_test_split(X, y, test_size=0.2, random_state=42)
            y_train = y_train_numeric >= y_train_numeric.describe(percentiles=[0.8])[5]
            y_test = y_test_numeric >= y_train_numeric.describe(percentiles=[0.8])[5]
            
            pipe.fit(X_train, y_train)
            train_metrics = calculate_metrics(y_train, pipe.predict(X_train))
            metric_data = update_metric_data(metric_data, train_metrics, feature_new, model, 'train')

            for i in range(n_iter_bootstrap):
                X_test_r, y_test_r = resample(X_test, y_test, random_state=np.random.randint(1000000))
                y_test_predict = pipe.predict(X_test_r)
                test_metrics_i = calculate_metrics(y_test_r, y_test_predict)
                metric_data = update_metric_data(metric_data, test_metrics_i, feature_new, model, 'test')
        importance = [0]*len(feature_names)
        if model == 'SVC':
            importance = pipe[-1].support_vectors_[0]
        elif model == 'RandomForestClassifier':
            importance = pipe[-1].feature_importances_
        elif model == 'LogisticRegression':
            importance = pipe[-1].coef_[0]
        else:
            continue
        feat_importance = pd.DataFrame({'feature':feature_names, 'importance': importance})
        feat_importance = feat_importance.sort_values(by=['importance'])
        print(feature_old, model)
        print(feat_importance.head(5))
        print(feat_importance.tail(5))

    metric_data['is_embedding'] = [True if 'embedding' in feature else False for feature in metric_data['feature']]
    figgen = create_metric_barplot_features_all_metric(metric_data)
    save_figs(figgen, 'f1_sorted')

    figgen = create_metric_barplot_models_all_metric(metric_data)
    save_figs(figgen, 'f1_sorted')

if __name__ == "__main__":
    main()
