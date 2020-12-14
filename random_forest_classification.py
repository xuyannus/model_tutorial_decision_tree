import numpy as np
import pandas as pd
import forestci as fci
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def load_data():
    data_df = pd.read_csv('data.csv')
    X = data_df.iloc[:, [2, 3]].values
    Y = data_df.iloc[:, 4].values
    return train_test_split(X, Y, test_size=0.25, random_state=0)


def cal_feature_importance(classifier, X_Test, Y_Test, plot=False):
    result = permutation_importance(classifier, X_Test, Y_Test, n_repeats=10, random_state = 0)
    print(result.importances_mean)

    if plot:
        sorted_idx = result.importances_mean.argsort()
        column_names = np.array(["Age", "EstimatedSalary", "Purchased"])
        fig, ax = plt.subplots()
        ax.boxplot(result.importances[sorted_idx].T, vert=False, labels = column_names[sorted_idx])
        ax.set_title("Permutation Importances (train set)")
        fig.tight_layout()
        plt.show()


def demo_variance_prediction(classifier, X_Train, X_Test):
    prediction_variance = fci.random_forest_error(classifier, X_Train, X_Test)
    print({
        "prediction_mean": classifier.predict(X_Test),
        "prediction_variance": prediction_variance
    })


def demo():
    X_Train, X_Test, Y_Train, Y_Test = load_data()
    # n_estimators needs to be tuned for the data size
    classifier = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
    classifier.fit(X_Train, Y_Train)

    print("=========training accuracy=========")
    print(confusion_matrix(Y_Train, classifier.predict(X_Train)))
    print("=========testing accuracy=========")
    print(confusion_matrix(Y_Test, classifier.predict(X_Test)))
    print("=========feature importance=========")
    cal_feature_importance(classifier, X_Test, Y_Test)
    print("=========variance=========")
    demo_variance_prediction(classifier, X_Train, X_Test)


# start the demo
demo()
