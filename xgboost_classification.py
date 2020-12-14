import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    data_df = pd.read_csv('data.csv')
    X = data_df.iloc[:, [2, 3]].values
    Y = data_df.iloc[:, 4].values
    return train_test_split(X, Y, test_size=0.25, random_state=0)


def demo():
    X_Train, X_Test, Y_Train, Y_Test = load_data()
    hyper_params = {'n_estimators': 200, 'max_depth': 20, 'learning_rate': 0.1, 'reg_lambda': 1}
    xg_model = xgb.XGBClassifier(**hyper_params)
    xg_model.fit(X_Train, Y_Train)

    print("=========training accuracy=========")
    print(confusion_matrix(Y_Train, xg_model.predict(X_Train)))
    print("=========testing accuracy=========")
    print(confusion_matrix(Y_Test, xg_model.predict(X_Test)))


# start the demo
demo()
