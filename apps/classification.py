import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix

def app():
    st.title("Classification Models")
    df_for_fs=pd.read_csv("dataforclassf.csv")
    y = df_for_fs.With_Kids
    X = df_for_fs.drop("With_Kids", axis=1)
    colnames = X.columns
    st.write("Features data table")
    st.write(X)
    st.write("Label data table (With_Kids)")
    st.write(y)

    def ranking(ranks, names, order=1):
        minmax = MinMaxScaler()
        ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        ranks = map(lambda x: round(x,2), ranks)
        return dict(zip(names, ranks))

    def plot_matrix(cm, classes, title):
        ax = sns.heatmap(cm, cmap="Blues", annot=True,fmt='1f', xticklabels=classes, yticklabels=classes, cbar=False)
        ax.set(title=title, xlabel="predicted label", ylabel="true label")

    with st.sidebar:
        st.info("Please Select which dataset to use SMOTE or UNSMOTE")
        selectionRadio = st.radio(
            "Dataset selection",
            ("SMOTE", "UNSMOTE"),index=0
        )

    model = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

    feat_selector = BorutaPy(
        verbose=0,
        estimator=model,
        n_estimators='auto',
        max_iter=100  # number of iterations to perform
    )

    feat_selector.fit(X.values, y.values.ravel())

    boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending = False)
    st.write('---------Top 10----------')
    st.write(boruta_score.head(10))
    st.write("---------Bottom 10----------")
    st.write(boruta_score.tail(10))
    st.write("Top 30 Boruta Features Model Accuracy")
    st.write('NB acc=  56.23565656565656')
    st.write('DT acc=  60.49838383838384')
    st.write('KNN acc=  56.446464646464655')
    st.write('SVM acc=  58.57484848484848')
    st.write('RFC acc=  58.503131313131306')
    st.write('LOR acc=  56.18535353535354')
    st.write("Accuracy Performance after using SMOTE dataset with default parameter")
    st.write("rf_SMOTE Accuracy Score: 0.652332361516035")
    st.write("svc_SMOTE Accuracy Score: 0.641399416909621")
    st.write("kn_SMOTE Accuracy Score: 0.6282798833819242")
    st.write("Accuracy Score: 0.6049562682215743")
    st.write("gnb_SMOTE Accuracy Score: 0.5954810495626822")
    st.write("")
    st.write("Running Gridsearch to get optimal hyperparameters for each of the models")

    if(selectionRadio=="SMOTE"):
        modelRadio = st.radio(
            "Model selection",
            ("Random Forest", "SVM","KNeighbours","Decision Tree","Ensemble"),index=0
        )
        if (modelRadio=='Random Forest'):
            rfhyperparam={'n_estimators': 1800, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 80, 'criterion': 'gini', 'bootstrap': False}
            st.write('Random forest optimal hyperparameter after GridSearch')
            st.write(rfhyperparam)
            st.write("Accuracy Score: 0.6399416909620991")
            st.write("ROC Score: 0.7403844722702405")
            st.write("Precision Score: 0.6455906821963394")
            st.write("F1 Score: 0.611023622047244")
            st.write("CV Score: [0.4        0.49180328 0.51094092 0.49781182 0.56345733]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[388, 312], [171, 501]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for Random Forest Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)
        elif(selectionRadio=='SVM'):
            svmhyperparam={'C':1 }
            st.write('SVM optimal hyperparameter after GridSearch')
            st.write(svmhyperparam)
            st.write("Accuracy Score: 0.6188046647230321")
            st.write("Precision Score: 0.6273381294964029")
            st.write("F1 Score: 0.6250896057347671")
            st.write("CV Score: [0.43169399 0.50491803 0.51531729 0.48796499 0.51203501]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[436, 264], [259, 413]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for SVM Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)
        elif(modelRadio=='KNeighbours'):
            rfhyperparam={'n_neighbors': 3}
            st.write('Random forest optimal hyperparameter after GridSearch')
            st.write(rfhyperparam)
            st.write("Accuracy Score: 0.6239067055393586")
            st.write("ROC Score: 0.6249404761904762")
            st.write("Precision Score: 0.6483870967741936")
            st.write("F1 Score: 0.6090909090909091")
            st.write("CV Score: [0.51584699 0.52896175 0.54595186 0.53829322 0.69256018]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[402, 298], [218, 454]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for KNeighbours Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)
        elif(modelRadio=='Decision Tree'):
            rfhyperparam={'ccp_alpha': 0.001, 'criterion': 'gini', 'max_depth': 9, 'max_features': 'auto'}
            st.write('Random forest optimal hyperparameter after GridSearch')
            st.write(rfhyperparam)
            st.write("Accuracy Score: 0.6275510204081632")
            st.write("ROC Score: 0.627827380952381")
            st.write("Precision Score: 0.6408345752608048")
            st.write("F1 Score: 0.6272793581327499")
            st.write("CV Score: [0.5715847  0.50710383 0.49343545 0.5        0.54595186]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[430, 270], [241, 431]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for Decision Tree Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)
        elif(modelRadio=='Ensemble'):
            results = pd.read_csv('ensembler1.csv')
            st.write("cart 0.689 (0.030)")
            st.write("knn 0.647 (0.021)")
            st.write("rf 0.678 (0.020)")
            st.write("stacking 0.681 (0.020)")
            st.write("Boxplot")
            fig = plt.figure(figsize = (10, 5))
            plt.boxplot(results, showmeans=True)
            st.pyplot(fig)
    if(selectionRadio=="UNSMOTE"):
        modelRadio = st.radio(
            "Model selection",
            ("Decision Tree", "Random Forest","KNeighbours","Decision Tree","Ensemble"),index=0
        )
        if (modelRadio=='Decision Tree'):
            rfhyperparam={'max_depth': 4, 'max_features': 0.4, 'min_samples_leaf': 0.04}
            st.write('Decision Tree optimal hyperparameter after GridSearch')
            st.write(rfhyperparam)
            st.write("Accuracy Score: 0.5833333333333334")
            st.write("ROC Score: 0.5757027819270207")
            st.write("Precision Score: 0.6160458452722063")
            st.write("F1 Score: 0.6323529411764706")
            st.write("CV Score: [0.5575  0.49125 0.52375 0.49    0.4825]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[430, 232], [268, 270]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for Decision Tree Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)

        elif(modelRadio=='Random Forest'):
            rfhyperparam={'n_estimators': 2000, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'criterion': 'gini', 'bootstrap': True}
            st.write('Decision Tree optimal hyperparameter after GridSearch')
            st.write(rfhyperparam)
            st.write("Accuracy Score: 0.5691666666666667")
            st.write("ROC Score: 0.64205171923842")
            st.write("Precision Score: 0.6124837451235371")
            st.write("F1 Score: 0.6456477039067855")
            st.write("CV Score: [0.6975  0.49625 0.4925  0.4925  0.46125]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[471, 219], [298, 212]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for Decision Tree Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)
        elif(modelRadio=='KNeighbours'):
            rfhyperparam={'n_neighbors': 8}
            st.write('Random forest optimal hyperparameter after GridSearch')
            st.write(rfhyperparam)
            st.write("Accuracy Score: 0.5641666666666667")
            st.write("ROC Score: 0.6209548167092924")
            st.write("Precision Score: 0.5947786606129398")
            st.write("F1 Score: 0.6670910248249522")
            st.write("CV Score: [0.60375 0.52875 0.50125 0.51    0.475  ]")
            st.write("Confusion matrix")
            fig = plt.figure(figsize = (10, 5))
            cm = np.array([[524, 166], [357, 153]])
            classes = ['no', 'yes']
            title = "Confusion Matrix for KNeighbours Classifier"
            plot_matrix(cm, classes, title)
            st.pyplot(fig)
        elif(modelRadio=='Ensemble'):
            results = pd.read_csv('ensembler2.csv')
            st.write("cart_UNSMOTE 0.427 (0.257)")
            st.write("knn_UNSMOTE 0.357 (0.034)")
            st.write("rf_UNSMOTE 0.478 (0.030)")
            st.write("stacking_UNSMOTE 0.611 (0.016)")
            st.write("Boxplot")
            fig = plt.figure(figsize = (10, 5))
            plt.boxplot(results, showmeans=True)
            st.pyplot(fig)