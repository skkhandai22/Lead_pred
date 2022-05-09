#import packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import Visualizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")

def finalModel(x_train, y_train,x_test):
    ct = ColumnTransformer(
        [('se', StandardScaler(), ['Total Time Spent on Website', 'Page Views Per Visit', 'TotalVisits'])],
        remainder='passthrough')
    random_forest = Pipeline([('transformer', ct), ('RandomForest',
                                                    RandomForestClassifier(n_estimators=300, min_samples_split=10,
                                                                           min_samples_leaf=2, max_features='auto',
                                                                           bootstrap=False, max_depth=None,
                                                                           random_state=42))])
    random_forest.fit(x_train, y_train)
    y_train_predict = random_forest.predict(x_train)
    y_test_predict = random_forest.predict(x_test)
    # print("Train Accuracy: %s" % (accuracy_score(y_train, y_train_predict)))
    # print("Test Accuracy: %s" % (accuracy_score(y_test, y_test_predict)))
    return accuracy_score(y_train, y_train_predict) ,accuracy_score(y_test, y_test_predict)


if __name__ == "__main__":
    # training and testing dataset
    train_dataset = "/home/mimansha/Downloads/lead-scoring-model-python-main/train.csv"
    test_dataset = "/home/mimansha/Downloads/lead-scoring-model-python-main/test.csv"

    train = pd.read_csv(train_dataset)
    train.drop("Unnamed: 0", axis=1, inplace=True)
    train.sample(frac=0.1).reset_index().drop("index", axis=1, inplace=True)
    x_train = train.drop("Converted", axis=1)
    y_train = train.iloc[:, -1]

    test = pd.read_csv(test_dataset)
    test.drop("Unnamed: 0", axis=1, inplace=True)
    test.sample(frac=0.1).reset_index().drop("index", axis=1, inplace=True)
    x_test = test.drop("Converted", axis=1)
    y_test = test.iloc[:, -1]

    tr, te = finalModel(x_train, y_train,x_test)
    print(tr,te)

    x_train.loc[:, "Actual Class"] = y_train
    x_train.loc[:, "Predicted Class"] = tr
    x_test.loc[:, "Actual Class"] = y_test
    x_test.loc[:, "Predicted Class"] = te

    predicted_df = x_train.append(x_test)

    plt.figure(figsize=(14, 6))
    plt.subplot(121)
    plt.title("Actual Class Label")
    sns.countplot(predicted_df["Actual Class"])
    plt.subplot(122)
    plt.title("Predicted Class Label")
    sns.countplot(predicted_df["Predicted Class"])





