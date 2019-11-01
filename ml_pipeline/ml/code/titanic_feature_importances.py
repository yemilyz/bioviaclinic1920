import os
from sklearn.externals import joblib
def main():
    joblib_file = os.path.join("results/titanic_Imputer_Scaler_RF_pipeline.pkl")
    pipe = joblib.load(joblib_file)
    forest = pipe.named_steps['RF']
    print(forest.feature_importances_)

if __name__ == "__main__":
    main()