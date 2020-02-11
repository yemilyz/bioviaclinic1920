import os
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',))
TRAINING_DATA_DIR = os.path.join(REPO_DIR, 'data')
FEATURE_DIR = TRAINING_DATA_DIR
EMBEDDING_FEATURE_DIR = os.path.join(FEATURE_DIR, 'embedding_features')

PROTPARAM_FEATURES = os.path.join(FEATURE_DIR, 'protparam_features.csv')

DI_LABELS_CSV = os.path.join(TRAINING_DATA_DIR, 'DI_out.csv')
