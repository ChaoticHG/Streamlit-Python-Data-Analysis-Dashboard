import math
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
def app():
    def viewState(data):
        states=df.City.unique()
        return states.astype(str)
    df = pd.read_csv('city.csv')
    st.title("Exploratory Data Analysis")
    st.write("We first add extra data points or features into the table")
    st.write("Datapoints added are")
    st.write("1. weather")
    st.write("2. day of the week")
    st.write("3. period of the day")
    st.write("4. is the date a holiday")
    st.write("5. City of the location")
    st.write("We replace the city information for some rows that have incorrect value")
    state=viewState(df)
    st.write(state)
    
    # Replacing some city names
    df['City'] = df['City'].replace(['Majlis Perbandaran Kajang', 'Majlis Perbandaran Ampang Jaya', 
                                 'Majlis Perbandaran Klang'], ['Kajang', 'Ampang Jaya', 'Klang'])
    state=viewState(df)
    st.write(state)

    st.write("The following is the null data count of the dataset")
    st.write(df.isnull().sum())
    df = df.astype({"Washer_No":"str","Dryer_No":"str"})
    st.write("Washer_No and Dryer_No is changed to categorical attribute")
    st.write("Hi")
    num_cols = ['Age_Range', 'TimeSpent_minutes', 'buyDrinks', 'TotalSpent_RM', 'Num_of_Baskets']
    fig = plt.figure(figsize = (10, 5))
    ax = sns.distplot(data=df, x='Age_Range',kde=True)
    plt.show()
    st.pyplot(fig)
    st.write("Hi")
    for col in num_cols:
        print(col)
        print('Skew :', round(df[col].skew(), 2))
        print('Kurt :', round(df[col].kurtosis(), 2))
        figx, ax = plt.subplots(figsize=(10, 10))
        sns.distplot(df[col], axlabel=col)
        st.pyplot(figx)
        #df[col].hist()
        figy, ax=plt.figure(figsize=(10, 10))
        plt.xlabel(col)
        ax.set_title("Total Sum of Loan vs Loan Amount")
        #plt.ylabel('count')
        df.boxplot(column = col)
        st.pyplot(figy)