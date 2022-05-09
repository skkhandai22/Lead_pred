import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# # import Visualizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# from lightgbm import LGBMClassifier
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression, SGDClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.naive_bayes import BernoulliNB, GaussianNB
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score, f1_score
import warnings
warnings.filterwarnings("ignore")
import pickle


def main():
    train = pd.read_csv("train.csv")
    print(train.shape)
    train.drop("Unnamed: 0", axis=1, inplace=True)
    train.sample(frac=0.1).reset_index().drop("index", axis=1, inplace=True)
    x_train = train.drop("Converted", axis=1)
    y_train = train.iloc[:, -1]
    pd.set_option('display.max_columns', 500)
    x_train.head()

    ct = ColumnTransformer(
        [('se', StandardScaler(), ['Total Time Spent on Website', 'Page Views Per Visit', 'TotalVisits'])],
        remainder='passthrough')
    return x_train,y_train,ct


## Final Model
def finalmodel():
    x_train, y_train, ct = main()
    random_forest = Pipeline([('transformer', ct), ('RandomForest',
                                                    RandomForestClassifier(n_estimators=300, min_samples_split=10,
                                                                           min_samples_leaf=2, max_features='auto',
                                                                           bootstrap=False, max_depth=None,
                                                                           random_state=42))])
    random_forest.fit(x_train, y_train)
    y_train_predict = random_forest.predict(x_train)
    print("Train Accuracy: %s" % (accuracy_score(y_train, y_train_predict)))
    x_train.loc[:, "Actual Class"] = y_train
    x_train.loc[:, "Predicted Class"] = y_train_predict

    pickle_out = open("modelrf.pkl", mode="wb")
    pickle.dump(random_forest, pickle_out)
    pickle_out.close()

finalmodel()