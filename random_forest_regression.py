import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection as xval
from sklearn.datasets import fetch_openml
import forestci as fci
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
    mpg_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    mpg_forest.fit(mpg_X_train, mpg_y_train)
    mpg_y_hat = mpg_forest.predict(mpg_X_test)

    print({
        "R2": r2_score(mpg_y_test, mpg_y_hat)
    })

    # Calculate the variance
    mpg_V_IJ_unbiased = fci.random_forest_error(mpg_forest, mpg_X_train, mpg_X_test)

    # Plot error bars for predicted MPG using unbiased variance
    plt.errorbar(mpg_y_test, mpg_y_hat, yerr=np.sqrt(mpg_V_IJ_unbiased), fmt='o')
    plt.plot([5, 45], [5, 45], 'k--')
    plt.xlabel('Reported MPG')
    plt.ylabel('Predicted MPG')
    plt.show()


# start the demo
demo()