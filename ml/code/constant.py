import os
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',))
TRAINING_DATA_DIR = os.path.join(REPO_DIR, 'data')
FEATURE_DIR = TRAINING_DATA_DIR
EMBEDDING_FEATURE_DIR = os.path.join(FEATURE_DIR, 'embedding_features')

PROTPARAM_FEATURES = os.path.join(FEATURE_DIR, 'protparam_features.csv')

SLIDING_WIN_FEATURES = os.path.join(FEATURE_DIR, 'sliding_win_0pad.csv')


DI_LABELS_CSV = os.path.join(TRAINING_DATA_DIR, 'DI_out.csv')


EMBEDDING_FEATURES_DIR = os.path.join(TRAINING_DATA_DIR, 'embedding_features')

EMBEDDING_5_7_FEATURES = os.path.join(EMBEDDING_FEATURES_DIR, 'feature_embedding_original_5_7.csv')
