import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, recall_score, precision_score, f1_score
import warnings
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve
warnings.filterwarnings("ignore")
from model_final import *
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


pickle_in = open('modelrf.pkl', 'rb')
classifier = pickle.load(pickle_in)

st.set_page_config(page_title='Lead Prediction Explorer',page_icon="logo.png",layout='wide',initial_sidebar_state='auto',)


def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test) #, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()

def test_data(datafile):
    test = pd.read_csv(datafile)
    test.drop("Unnamed: 0", axis=1, inplace=True)
    test.sample(frac=0.1).reset_index().drop("index", axis=1, inplace=True)
    x_test = test.drop("Converted", axis=1)
    y_test = test.iloc[:, -1]
    return x_test, y_test



if __name__ == '__main__':
    st.markdown(
        '<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
        unsafe_allow_html=True)


    col1, col2, col3 = st.columns([1, 6, 1])
    st.markdown("""
    <nav class ="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #FF4B$B;">
    """,unsafe_allow_html=True)

    with col1:
        st.sidebar.image('Compunnel-Digital-Logo.png', width=125)
    st.sidebar.title('''**Lead Prediction Explorer**''')
    st.markdown("""<style>[data-testid="stSidebar"][aria-expanded="true"]
    > div:first-child {width: 450px;}[data-testid="stSidebar"][aria-expanded="false"]
    > div:first-child {width: 450px;margin-left: -400px;}</style>""",
    unsafe_allow_html=True)

    uploaded_files = st.sidebar.file_uploader("Upload Data File", type=['csv'], accept_multiple_files=False)
    print(uploaded_files)
    if uploaded_files:
        selected = option_menu(
            menu_title="",
            options=["Actual Dataset","After EDA : Dataset", "Prediction & Visualization"],
            icons=["house", "clipboard-data"],
            orientation="horizontal"
        )
        if selected == "Actual Dataset":
            st.title("The actual data parameters shown as below:")
            data = pd.read_csv("Lead Scoring.csv")
            st.dataframe(data)

        if selected == "After EDA : Dataset":
            st.title("The data parameters shown as below:")
            data = pd.read_csv(uploaded_files.name)
            data.drop("Unnamed: 0", axis=1, inplace=True)
            data.sample(frac=0.1).reset_index().drop("index", axis=1, inplace=True)
            st.dataframe(data)

        if selected == "Prediction & Visualization":
            x_test, y_test = test_data(uploaded_files.name)
            x_train, y_train, cf = main()
            test_pred = classifier.predict(x_test)
            train_pred = classifier.predict(x_train)
            score_test = accuracy_score(y_test, test_pred)
            score_train = accuracy_score(y_train,train_pred)
            f1_test = f1_score(y_test, test_pred)
            f1_train = f1_score(y_train,train_pred)
            cm = confusion_matrix(y_test, test_pred)

            train_cf_report = pd.DataFrame(classification_report(y_train, train_pred, output_dict=True))
            test_cf_report = pd.DataFrame(classification_report(y_test, test_pred, output_dict=True))
            # print(score_,cm,f1)

            x_train.loc[:, "Actual Class"] = y_train
            x_train.loc[:, "Predicted Class"] = train_pred
            x_test.loc[:, "Actual Class"] = y_test
            x_test.loc[:, "Predicted Class"] = test_pred

            predicted_df = x_train.append(x_test)
            fig1, axes = plt.subplots(1, 2,figsize=(16,9))
            axes[0].set_title("Actual Class Label")
            sns.countplot(predicted_df["Actual Class"],ax=axes[0])
            axes[1].set_title("Predicted Class Label")
            sns.countplot(predicted_df["Predicted Class"],ax=axes[1])
            st.pyplot(fig1)

            # st.write("Training Accuracy Score:",score_train)
            st.write("Testing Accuracy Score:",score_test)

            # st.write("F1score_training:",f1_train)
            st.write("F1score_testing:", f1_test)

            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(classifier, x_test, y_test)  # , display_labels=class_names)
            st.pyplot()
            st.write("Confusion matrix is the way to evaluate model's performance. It basically compared the predictions with actual labels and divide them into four quadrants.")

            st.subheader("ROC Curve")
            plot_roc_curve(classifier, x_test, y_test)
            st.pyplot()
            st.write("ROC Curves summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.")


            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(classifier, x_test, y_test)
            st.pyplot()
            st.write("Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.")

            # plt.figure(figsize=(16, 6))
            # ax1 = plt.subplot(121)
            # rf_disp = plot_precision_recall_curve(classifier, x_test, y_test, ax=ax1,
            #                                       name='Random Forest Precision Recall Curve')
            # st.pyplot(rf_disp)
