import os 
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# globals to help configure what metric to sort by
global F1, ROCAUC, sort_mapping_master
F1 = 'f1'
ROCAUC = 'roc_auc'
sort_mapping_master = {
    F1: {'ranking':'rank_test_f1', 'score':'mean_test_f1', 'metric':'f1'},
    ROCAUC: {'ranking':'rank_test_roc_auc', 'score':'mean_test_roc_auc', 'metric':'rocauc'},
    }

def get_ranked_metrics(sort_mapping, n_splits=10):
    score = sort_mapping['score']
    ranking = sort_mapping['ranking']
    best_metrics_all = pd.DataFrame(columns=[])
    for root, dirs, _ in os.walk('result_{}splits/'.format(n_splits), topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, '*cv_metrics.csv'))
            for metric_file in files_in_dir:
                metric = pd.read_csv(metric_file)
                metric = metric.loc[metric[ranking]==1]
                descriptor = os.path.split(metric_file)[-1].split('.')[0]
                is_embedding = "embedding" in descriptor
                metric['descriptor'] = descriptor
                metric['is_embedding'] = is_embedding
                if best_metrics_all.empty:
                    best_metrics_all = metric
                else:
                    best_metrics_all = best_metrics_all.append(metric)
    best_metrics_all = best_metrics_all.sort_values(by=['is_embedding', score], ascending=False)
    return best_metrics_all

def tranform_metrics_to_long(metric_data):
    """ takes wide form metric data and returns long form wide data, as well as a sorted
    list of features (best to worst)
    """
    new_descriptor = metric_data["descriptor"].str.rsplit("_", n = 3, expand = True)
    is_embedding = metric_data["is_embedding"]
    metric_data = metric_data.filter(regex='split')
    value_list = list(metric_data)
    # print(value_list)

    metric_data['feature'] = new_descriptor[0]
    model = new_descriptor[1].str.replace('Dummy', '*Dummy')
    metric_data['model'] = model
    metric_data['is_embedding'] = is_embedding
    metric_data_long = pd.melt(
        metric_data,
        id_vars=['feature', 'model', 'is_embedding'],
        value_vars=value_list,
        value_name='score',
        var_name='metric'
        )
    set_metric = metric_data_long['metric'].str.rsplit("_", n=1, expand=True)
    new_metric = set_metric[1].str.replace('auc', 'rocauc')
    new_set = set_metric[0].str.split("_", n=3, expand=True)[1]
    metric_data_long['metric'] = new_metric
    metric_data_long['set'] = new_set
    return metric_data_long

def get_ranked_embedding_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==True].feature.unique()

def get_ranked_pc_features(metric_data_long):
    return metric_data_long[metric_data_long['is_embedding']==False].feature.unique()

def main():
    sorted_by = F1
    sort_mapping = sort_mapping_master[sorted_by]

    metric_data = get_ranked_metrics(sort_mapping) # replace this line with pd.read_csv('metric_data_example.csv')
    metric_data_long = tranform_metrics_to_long(metric_data)

    embedding_features = get_ranked_embedding_features(metric_data_long)
    pc_features = get_ranked_pc_features(metric_data_long)

    top_features = embedding_features[:3].tolist() + pc_features[:3].tolist()
    print(top_features)


if __name__ == "__main__":
    main()

