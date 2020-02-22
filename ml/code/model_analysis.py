import pickle
from sklearn.model_selection import train_test_split
from joblib import dump, load

import pandas as pd

from datasets import get_dataset
import preprocessors as preprocessors



# model_path = 'result/feature_protparam/model/feature_protparam_RF_pipeline.pkl'
# feature_path = 'data/feature_protparam.csv'

model_path = 'result/feature_sliding_win_0pad/model/feature_sliding_win_0pad_RF_pipeline.pkl'
feature_path = 'data/feature_sliding_win_0pad.csv'

X, y, featue_names = get_dataset(feature_path)
X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)
pipe = load(model_path)

pipe = pipe.fit(X_train, y_train)

clf = pipe['RF']
clf.feature_importances_

feature_importance = pd.DataFrame({'feature_names': featue_names, 'feature_importance': clf.feature_importances_})
feature_importance = feature_importance.sort_values(by=['feature_importance'], ascending=False)


score = pipe.score(X_train, y_train)