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

# def evaluate_model(model, x_train, y_train, x_test, y_test):
def evaluate_model(model, x_train, y_train):
    model = model.fit(x_train, y_train)
    predict_train_y = model.predict(x_train)
    # predict_test_y = model.predict(x_test)

    print("**Accuracy Score**")
    train_accuracy = accuracy_score(y_train, predict_train_y)
    # test_accuracy = accuracy_score(y_test, predict_test_y)
    print("Train Accuracy is: %s" % (train_accuracy))
    # print("\nTest Accuracy is: %s" % (test_accuracy))
    print("---------------------------------------------------------")

    print("\n**Accuracy Error**")
    train_error = (1 - train_accuracy)
    # test_error = (1 - test_accuracy)
    print("Train Error: %s" % (train_error))
    # print("\nTest Error: %s" % (test_error))
    print("---------------------------------------------------------")

    print("\n**Classification Report**")
    train_cf_report = pd.DataFrame(classification_report(y_train, predict_train_y, output_dict=True))
    # test_cf_report = pd.DataFrame(classification_report(y_test, predict_test_y, output_dict=True))
    print("Train Classification Report:")
    print(train_cf_report)
    # print("\n Test Classification Report:")
    # print(test_cf_report)
    print("---------------------------------------------------------")

    print("\n**Confusion Matrix**")
    train_conf = confusion_matrix(y_train, predict_train_y)
    # test_conf = confusion_matrix(y_test, predict_test_y)
    print("Train Confusion Matrix Report:")
    print((train_conf))
    # print("\n Test Confusion Matrix Report:")
    # print((test_conf))


def trainModel():
    x_train, y_train, ct = main()
    random_forest_pipeline = Pipeline([('transformer', ct), ('RandomForest', RandomForestClassifier(random_state=42))])
    adaboost_pipeline = Pipeline([('transformer', ct), ('Adaboost', AdaBoostClassifier(random_state=42))])
    ExtraTree_pipeline = Pipeline([('transformer', ct), ('ExtraTreeClassifier', ExtraTreesClassifier(random_state=42))])
    BaggingClassifier_pipeline = Pipeline([('transformer', ct), (
    'BaggingClassifier', BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42))])
    GradientBoost_pipeline = Pipeline(
        [('transformer', ct), ('GradientBoosting', GradientBoostingClassifier(random_state=42))])
    dtree_pipeline = Pipeline([('transformer', ct), ('DecisionTree', DecisionTreeClassifier(random_state=42))])
    knn_pipeline = Pipeline([('transformer', ct), ('KNN', KNeighborsClassifier())])
    lr_pipeline = Pipeline([('transformer', ct), ('LogisticRegression', LogisticRegression(random_state=42))])
    sgd_pipeline = Pipeline([('transformer', ct), ('StochasticGradient', SGDClassifier(random_state=42))])
    mlp_pipeline = Pipeline([('transformer', ct), ('MLPClassifier', MLPClassifier(random_state=42))])
    naive_pipeline = Pipeline([('transformer', ct), ('NaiveBayes', GaussianNB())])
    svc_pipeline = Pipeline([('transformer', ct), ('SVM', SVC(random_state=42))])
    lightgbm_pipeline = Pipeline([('transformer', ct), ('lightgbm', LGBMClassifier(random_state=42))])
    catboost_pipeline = Pipeline([('transformer', ct), ('CatBoost', CatBoostClassifier(random_state=42, silent=True))])
    xgboost_pipeline = Pipeline([('transformer', ct), ('XGBoost', XGBClassifier(random_state=42))])

    pipeline_list = [random_forest_pipeline, adaboost_pipeline, ExtraTree_pipeline, BaggingClassifier_pipeline,
                     GradientBoost_pipeline,
                     dtree_pipeline, knn_pipeline, lr_pipeline, sgd_pipeline, mlp_pipeline, naive_pipeline, svc_pipeline,
                     lightgbm_pipeline, catboost_pipeline, xgboost_pipeline]

    pipe_dict = {0: "RandomForest", 1: "Adaboost", 2: "ExtraTree", 3: "BaggingClassifier", 4: "GradientBoosting",
                 5: "DecisionTree",
                 6: "KNN", 7: "Logistic", 8: "SGD Classifier", 9: "MLPClassifier", 10: "NaiveBayes",
                 11: "SVM", 12: "LightGBM", 13: "Catboost", 14: "XGBoost"}

    lst = []
    for idx, pipe in enumerate(pipeline_list):
        score = cross_val_score(pipe, x_train, y_train, cv=10, scoring='accuracy')
        print(pipe_dict[idx], ":", score.mean())

    ### RANDOM FOREST CLASSIFIER
    rforest = RandomForestClassifier(random_state=42)
    evaluate_model(rforest, x_train, y_train)

    ### GRADIENT BOOSTING CLASSIFIER
    GradientBoost = GradientBoostingClassifier(random_state=42)
    evaluate_model(GradientBoost, x_train, y_train)

    ### LIGHTGBM CLASSIFIER
    lgbm = LGBMClassifier(random_state=42)
    evaluate_model(lgbm, x_train, y_train)

    ### CATBOOST CLASSIFIER
    catboost_classif = CatBoostClassifier(random_state=42, silent=True)
    evaluate_model(catboost_classif, x_train, y_train)

    ### Random Forest Hyperparameter Tuning
    new_pipeline = Pipeline([('transformer', ct), ('classifier', RandomForestClassifier(random_state=42))])
    rf_params = [{
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': np.arange(100, 2000, 200),
        'classifier__max_depth': [None, 10, 20, 30, 50, 70, 80, 100],
        'classifier__min_samples_split': [2, 3, 5, 7, 10],
        'classifier__min_samples_leaf': [1, 2, 3, 4, 5, ],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__bootstrap': [True, False]
    }]
    # rf_param_grid
    random_search = RandomizedSearchCV(estimator=new_pipeline, param_distributions=rf_params, scoring='accuracy', n_jobs=-1,
                                       cv=10, random_state=42)
    best_rf_model = random_search.fit(x_train, y_train)
    # best_rf_model.best_params_
    print("Best Score: %s" % (best_rf_model.best_score_))
    # best_rf_model.best_estimator_
    rf_classif_pipeline = Pipeline([('transformer', ct), ('RandomForest',
                                                          RandomForestClassifier(n_estimators=300, min_samples_split=10,
                                                                                 min_samples_leaf=2, max_features='auto',
                                                                                 bootstrap=False, max_depth=None,
                                                                                 random_state=42))])
    rf_classif_pipeline.fit(x_train, y_train)
    #save the trained model in pickl file
    # test_prediction = rf_classif_pipeline.predict(x_test)
    # accuracy_score(y_test, test_prediction)


    #test data workaround
    def check_metric(y_test, y_predict):
        print("**Accuracy Score**")
        test_accuracy = accuracy_score(y_test, y_predict)
        print("\nTest Accuracy is: %s" % (test_accuracy))
        print("---------------------------------------------------------")

        print("\n**Accuracy Error**")
        test_error = (1 - test_accuracy)
        print("\nTest Error: %s" % (test_error))
        print("---------------------------------------------------------")

        print("\n**Classification Report**")
        test_cf_report = pd.DataFrame(classification_report(y_test, y_predict, output_dict=True))
        print("\n Test Classification Report:")
        print(test_cf_report)
        print("---------------------------------------------------------")

        print("\n**Confusion Matrix**")
        test_conf = confusion_matrix(y_test, y_predict)
        print("\n Test Confusion Matrix Report:")
        print((test_conf))


    # check_metric(y_test, test_prediction)

    ### Cat Boost Hyperparameter Tuning
    cb_new_pipeline = Pipeline([('transformer', ct), (
    'classifier', CatBoostClassifier(random_state=42, task_type='CPU', silent=True, eval_metric='accuracy'))])
    catboost_params = [{
        'classifier': [CatBoostClassifier()],
        'classifier__iterations': [10],
        'classifier__learning_rate': [0.0001, 0.001, 0.003, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        'classifier__depth': [2, 4, 6, 8, 10, 12],
        'classifier__l2_leaf_reg': [2, 3, 5, 7, 9, 11, 12, 15, 18, 20, 25, 27],
        'classifier__random_strength': [1],
        'classifier__border_count': [50, 100, 150, 200, 254],
    }]
    cb_random_search = RandomizedSearchCV(estimator=cb_new_pipeline, param_distributions=catboost_params,
                                          scoring='accuracy', n_jobs=-1, cv=10, random_state=42)
    cb_random_search.fit(x_train, y_train)
    # cb_random_search.best_params_
    # cb_random_search.best_score_
    catboost_model = CatBoostClassifier(random_strength=1, learning_rate=0.5, l2_leaf_reg=7, iterations=10, depth=10,
                                        border_count=50,
                                        silent=True, eval_metric='Accuracy', task_type='CPU')
    catboost_model.fit(x_train, y_train, silent=True, plot=True)
    # y_test_predict = catboost_model.predict(x_test)
    # check_metric(y_test, y_test_predict)

    ## Final Model
    random_forest = Pipeline([('transformer', ct), ('RandomForest',
                                                    RandomForestClassifier(n_estimators=300, min_samples_split=10,
                                                                           min_samples_leaf=2, max_features='auto',
                                                                           bootstrap=False, max_depth=None,
                                                                           random_state=42))])

    random_forest.fit(x_train, y_train)
    y_train_predict = random_forest.predict(x_train)
    # y_test_predict = random_forest.predict(x_test)
    print("Train Accuracy: %s" % (accuracy_score(y_train, y_train_predict)))
    # print("Test Accuracy: %s" % (accuracy_score(y_test, y_test_predict)))
    x_train.loc[:, "Actual Class"] = y_train
    x_train.loc[:, "Predicted Class"] = y_train_predict
    # x_test.loc[:, "Actual Class"] = y_test
    # x_test.loc[:, "Predicted Class"] = y_test_predict

    # predicted_df = x_train.append(x_test)
    #
    # plt.figure(figsize=(14, 6))
    # plt.subplot(121)
    # plt.title("Actual Class Label")
    # sns.countplot(predicted_df["Actual Class"])
    # plt.subplot(122)
    # plt.title("Predicted Class Label")
    # sns.countplot(predicted_df["Predicted Class"])