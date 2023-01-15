import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
from apyori import apriori

def app():
    st.title("Association Rule Mining")
    df = pd.read_csv("combinedrecord.csv", index_col=0)
    df['buyDrinks'] = df['buyDrinks'].fillna(0).astype(int)
    st.write("buyDrinks columns row with NaN are replaced with 0")
    cols = ['Washer_No','Dryer_No','buyDrinks']
    df_ass = df[cols]
    st.write(df_ass)
    st.write("Result after running Association Rule Mining")
    df = df.astype({"Washer_No":"str","Dryer_No":"str"})
    #replace blanks with 0
    df['buyDrinks'] = df['buyDrinks'].fillna(0).astype(int)
    #create new dummy dataset
    cols = ['Washer_No','Dryer_No','buyDrinks']
    washerdict={
        '3' : "W3",
        '4' : "W4",
        '5' : "W5",
        '6' : "W6",
    }

    dryerdict={
        '7' : "D7",
        '8' : "D8",
        '9' : "D9",
        '10' : "D10",
    }

    buyDrinksdict={
        0 : "B0",
        1 : "B1",
        2 : "B2",
        3 : "B3",
        4 : "B4",
        5 : "B5"
    }
    df_ass=pd.DataFrame()
    df_ass['Washer_No']=df['Washer_No'].replace(washerdict)
    df_ass['Dryer_No']=df['Dryer_No'].replace(dryerdict)
    df_ass['buyDrinks']=df['buyDrinks'].replace(buyDrinksdict)
    st.write("Data column after preprocessing")
    st.write(df_ass)

    records = []
    # loop to group data
    for i in range(0, len(df_ass)):
        records.append([str(df_ass.values[i,j])for j in range (0,3)])

    #apply Association Rule Mining
    association_results = list(apriori(records, min_support=0.0045,
                            min_confidence=0.2,
                            min_lift=1,
                            min_length=2))
    #print(association_results)
    dfapr = []
    count=0
    for item in association_results[8:]:
        count +=1
        pair = item[0],
        items=[x for x in pair]
        #print(items)
        rule = "(Rule " + str(count) + ") " + str(items)
        support = str(round(item[1],3))
        confidence= str(round(item[2][0][2],4))
        lift = str(round(item[2][0][3],4))
        dfapr.append([rule,support,confidence,lift])
    dfapr=pd.DataFrame.from_records(dfapr,columns=['Rule','Support','Confidence','Lift'])
    dfapr.sort_values(by=['Support'],ascending=False).head(10)
    dfapr.sort_values(by=['Confidence'],ascending=False).head(10)
    st.write("Top 10 association rule result")
    st.write(dfapr.sort_values(by=['Lift'],ascending=False).head(10))
