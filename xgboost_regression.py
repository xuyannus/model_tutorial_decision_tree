import numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection as xval
from sklearn.datasets import fetch_openml
import xgboost as xgb
from sklearn.metrics import r2_score


def load_data():
    # retreive mpg data from machine learning library
    mpg_data = fetch_openml('autompg')

    # separate mpg data into predictors and outcome variable
    mpg_X = mpg_data["data"]
    mpg_y = mpg_data["target"]

    # remove rows where the data is nan
    not_null_sel = np.invert(np.sum(np.isnan(mpg_data["data"]), axis=1).astype(bool))
    mpg_X = mpg_X[not_null_sel]
    mpg_y = mpg_y[not_null_sel]

    # split mpg data into training and test set
    return xval.train_test_split(mpg_X, mpg_y, test_size=0.25, random_state=42)


def demo():
    mpg_X_train, mpg_X_test, mpg_y_train, mpg_y_test = load_data()
    hyper_params = {'objective': 'reg:squarederror', 'n_estimators': 200, 'max_depth': 20, 'learning_rate': 0.1, 'reg_lambda': 1}
    xg_reg = xgb.XGBRegressor(**hyper_params)

    xg_reg.fit(mpg_X_train, mpg_y_train)
    mpg_y_hat = xg_reg.predict(mpg_X_test)
    print({
        "R2": r2_score(mpg_y_test, mpg_y_hat)
    })


# start the demo
demo()