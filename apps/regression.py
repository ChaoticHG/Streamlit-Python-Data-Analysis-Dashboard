import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns

def app():
    st.title("Regression Models")
    with st.sidebar:
        st.info("Please Select Which Model To View")
        selectionRadio = st.radio(
            "Model List",
            ("Linear Regression", "XGBoost Regression"),index=0
        )
    df = pd.read_csv('processed.csv')
    if(selectionRadio=="Linear Regression"):
        st.subheader(selectionRadio)
        selectionBox=st.selectbox(
            'Feature Selection Select',
            ("Boruta Features", "RFE"),index=0
        )
        if(selectionBox=="Boruta Features"):
            selectFeature = st.select_slider(
            'Select number of Boruta Feature',
            options=[10,20,30])
            if(selectFeature==10):
                st.write("Poly_mse for our testing dataset with tuning is : 0.9834999650776531")
                st.write("Poly_rmse for our testing dataset with tuning is : 0.9917156674559766")
            elif(selectFeature==20):
                st.write("Poly_mse for our testing dataset with tuning is : 1.0782652357834224")
                st.write("Poly_rmse for our testing dataset with tuning is : 1.038395510286626")
            elif(selectFeature==30):
                st.write("Poly_mse for our testing dataset with tuning is : 0.9808727629677613")
                st.write("Poly_rmse for our testing dataset with tuning is : 0.9903902074272348")
        elif(selectionBox=="RFE"):
            st.write("Poly_mse for our testing dataset with tuning is : 0.9661663473515383")
            st.write("Poly_rmse for our testing dataset with tuning is : 0.982937611118599")
            fsedata=pd.read_csv('fselinear.csv')
            fig = plt.figure()
            sns.distplot(fsedata, bins = 20)
            fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
            plt.xlabel('Errors', fontsize = 18) 
            st.pyplot(fig)
    elif(selectionRadio=="XGBoost Regression"):
        st.subheader(selectionRadio)
        selectionBox=st.selectbox(
            'Feature Selection Select',
            ("Boruta Features", "RFE"),index=0
        )
        if(selectionBox=="Boruta Features"):
            selectFeature = st.select_slider(
            'Select number of Boruta Feature',
            options=[10,20,30])
            if(selectFeature==10):
                st.write("Poly_mse for our testing dataset with tuning is : 1.0316180645792983")
                st.write("Poly_rmse for our testing dataset with tuning is : 1.015686006883672")
            elif(selectFeature==20):
                st.write("Poly_mse for our testing dataset with tuning is : 1.0183730525153984")
                st.write("Poly_rmse for our testing dataset with tuning is : 1.0091447133664222")
            elif(selectFeature==30):
                st.write("Poly_mse for our testing dataset with tuning is : 1.006762205700317")
                st.write("Poly_rmse for our testing dataset with tuning is : 1.0033754061667632")
        elif(selectionBox=="RFE"):
            st.write("Poly_mse for our testing dataset with tuning is : 0.9661663473515383")
            st.write("Poly_rmse for our testing dataset with tuning is : 0.982937611118599")
            fsedata=pd.read_csv('fselinear.csv')
            fig = plt.figure()
            sns.distplot(fsedata, bins = 20)
            fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
            plt.xlabel('Errors', fontsize = 18) 
            st.pyplot(fig)


