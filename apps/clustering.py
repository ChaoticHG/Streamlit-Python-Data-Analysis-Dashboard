import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns


def app():
    st.title("Clustering Analysis")
    selectionRadio = st.radio(
            "Number of n cluster",
            ("2", "3","4"),index=0
        )

    st.write('Elbow Method:')
    cost=pd.read_csv('cost.csv', squeeze=True,header=None, skiprows=1)
    cost = cost.drop(cost.columns[0], axis=1)
    cost = cost.values.tolist()
    fig = plt.figure(figsize = (10, 5))
    plt.plot(range(1,10), cost, marker='o')
    st.pyplot(fig)
    if(selectionRadio=="2"):
        df_clustered =pd.read_csv('cluster1.csv')
        fig = plt.figure(figsize=(13,17))
        fig=sns.pairplot(df_clustered, hue="cluster")
        st.pyplot(fig)
    elif(selectionRadio=="3"):
        df_clustered =pd.read_csv('cluster2.csv')
        fig = plt.figure(figsize=(13,17))
        fig=sns.pairplot(df_clustered, hue="cluster")
        st.pyplot(fig)
    elif(selectionRadio=="4"):
        df_clustered =pd.read_csv('cluster3.csv')
        fig = plt.figure(figsize=(13,17))
        fig=sns.pairplot(df_clustered, hue="cluster")
        st.pyplot(fig)