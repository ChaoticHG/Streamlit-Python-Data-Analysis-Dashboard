import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import base64
from fpdf import FPDF
from tempfile import NamedTemporaryFile
def app():
    def create_download_link(val, filename):
        b64 = base64.b64encode(val)  # val looks like b'...'
        return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'
    def viewState(data):
        states=df.City.unique()
        return states.astype(str)
    texts=[]
    figs = []
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
    st.write("Bapak Dia Project Ni Gempak Giler Bossku")
    num_cols = ['Age_Range', 'TimeSpent_minutes', 'buyDrinks', 'TotalSpent_RM', 'Num_of_Baskets']
    for col in num_cols:
        df[col].fillna(np.round(df[col].mean()), inplace=True)
    for col in num_cols:
        st.write(col)
        st.write('Skew :', round(df[col].skew(), 2))
        st.write('Kurt :', round(df[col].kurtosis(), 2))
        fig= plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        ax = sns.distplot(df[col], x=df[col],kde=True)
        #df[col].hist()
        plt.subplot(1, 2, 2)
        ax.set_title("Total Sum of Loan vs Loan Amount")
        #plt.ylabel('count')
        df.boxplot(column = col)
        figs.append(fig)
        st.pyplot(fig)
    
    fig, axes = plt.subplots(7, 3, figsize = (22, 70))
    st.write('Bar plot for all categorical variables in the dataset')
    sns.countplot(ax = axes[0, 0], x = 'Race', data = df, color = 'blue', 
                order = df['Race'].value_counts().index)
    sns.countplot(ax = axes[0, 1], x = 'Gender', data = df, color = 'blue', 
                order = df['Gender'].value_counts().index)
    sns.countplot(ax = axes[0, 2], x = 'Body_Size', data = df, color = 'blue', 
                order = df['Body_Size'].value_counts().index)
    sns.countplot(ax = axes[1, 0], x = 'With_Kids', data = df, color = 'blue', 
                order = df['With_Kids'].value_counts().index)
    sns.countplot(ax = axes[1, 1], x = 'Kids_Category', data = df, color = 'blue', 
                order = df['Kids_Category'].value_counts().index)
    sns.countplot(ax = axes[1, 2], x = 'Basket_Size', data = df, color = 'blue', 
                order = df['Basket_Size'].value_counts().index)
    sns.countplot(ax = axes[2, 0], x = 'Basket_colour', data = df, color = 'blue', 
                order = df['Basket_colour'].value_counts().index)
    sns.countplot(ax = axes[2, 1], x = 'Attire', data = df, color = 'blue', 
                order = df['Attire'].value_counts().index)
    sns.countplot(ax = axes[2, 2], x = 'Shirt_Colour', data = df, color = 'blue', 
                order = df['Shirt_Colour'].value_counts().index)
    sns.countplot(ax = axes[3, 0], x = 'shirt_type', data = df, color = 'blue', 
                order = df['shirt_type'].value_counts().index)
    sns.countplot(ax = axes[3, 1], x = 'Pants_Colour', data = df, color = 'blue', 
                order = df['Pants_Colour'].value_counts().index)
    sns.countplot(ax = axes[3, 2], x = 'pants_type', data = df, color = 'blue', 
                order = df['pants_type'].value_counts().index)
    sns.countplot(ax = axes[4, 0], x = 'Wash_Item', data = df, color = 'blue', 
                order = df['Wash_Item'].value_counts().index)
    sns.countplot(ax = axes[4, 1], x = 'Washer_No', data = df, color = 'blue', 
                order = df['Washer_No'].value_counts().index)
    sns.countplot(ax = axes[4, 2], x = 'Dryer_No', data = df, color = 'blue', 
                order = df['Dryer_No'].value_counts().index)
    sns.countplot(ax = axes[5, 0], x = 'Spectacles', data = df, color = 'blue', 
                order = df['Spectacles'].value_counts().index)
    sns.countplot(ax = axes[5, 1], x = 'Is_Holiday', data = df, color = 'blue', 
                order = df['Is_Holiday'].value_counts().index)
    sns.countplot(ax = axes[5, 2], x = 'Period', data = df, color = 'blue', 
                order = df['Period'].value_counts().index)
    sns.countplot(ax = axes[6, 0], x = 'Week_Day', data = df, color = 'blue', 
                order = df['Week_Day'].value_counts().index)
    sns.countplot(ax = axes[6, 1], x = 'weather', data = df, color = 'blue', 
                order = df['weather'].value_counts().index)
    sns.countplot(ax = axes[6, 2], x = 'City', data = df, color = 'blue', 
                order = df['City'].value_counts().index)
    axes[2][0].tick_params(labelrotation=45)
    axes[2][2].tick_params(labelrotation=45)
    axes[3][1].tick_params(labelrotation=45)
    axes[6][1].tick_params(labelrotation=60)
    axes[5][2].tick_params(labelrotation=45)
    axes[6][2].tick_params(labelrotation=45)
    figs.append(fig)
    st.pyplot(fig)
    st.write("Exploratory OMG")
    st.write("Q1: Are there more customers visiting during weekends as compared to weekdays?")
    st.write("Sunday and Saturday definitely have more customers visiting.")
    st.write("Bivariate Analysis - Numerical vs Numerical")
    st.write("Q2: Is there any relationship between the numerical features?")
    fig = plt.figure(figsize=(13,17))
    fig=sns.pairplot(data=df)
    figs.append(fig)
    st.pyplot(fig)
    st.write("Correlation Between Variables")
    # heatmap
    fig=plt.figure(figsize=(12, 7))
    corr = df.corr()

    # Getting the Upper Triangle of the co-relation matrix
    matrix = np.triu(np.ones_like(corr))

    # using the upper triangle matrix as mask 
    sns.heatmap(corr, annot=True, mask=matrix)
    figs.append(fig)
    st.pyplot(fig)

    
    st.write("From the scatterplots, there are no relationship between any of the numerical features. It is evidenced that we can't see any positive or negative correlations in the scatterplots. The correlation values calculated reflect the same findings.")
    st.write("Bivariate Analysis")
    st.write("Q3: Is there any relationship between basket size and race?")
    fig=plt.figure(figsize=(12, 7))
    ax = sns.countplot(data=df, x='Race', hue='Basket_Size')
    sns.move_legend(ax, bbox_to_anchor=(1, 1.02), loc='upper left')

    for c in ax.containers:
        # set the bar label
        ax.bar_label(c, label_type='center')
    figs.append(fig)
    st.pyplot(fig)
    st.write("p value is 0.12454255644416322")
    st.write("Independent (H0 holds true, the variables do not have a significant relation.)")
    st.write("From the grouped bar chart, every race tends to use big basket size. However, there is no significant relationship between 'Race' and 'Basket_Size' after chi-square is calculated.")
    st.write("Q4: Did weather information impact the sales?")
    fig = plt.figure(figsize=(13,17))
    ax = sns.boxplot(data = df, x = 'weather', y = 'TotalSpent_RM')
    ax.tick_params(labelrotation=90)
    figs.append(fig)
    st.pyplot(fig)
    st.write(stats.f_oneway(df['TotalSpent_RM'][df['weather'] == 'Partly cloudy'],
               df['TotalSpent_RM'][df['weather'] == 'Broken clouds'],
               df['TotalSpent_RM'][df['weather'] == 'Passing clouds'],
              df['TotalSpent_RM'][df['weather'] == 'Partly sunny'],
              df['TotalSpent_RM'][df['weather'] == 'Dense fog'],
              df['TotalSpent_RM'][df['weather'] == 'Thunderstorms'],
              df['TotalSpent_RM'][df['weather'] == 'Light rain'],
              df['TotalSpent_RM'][df['weather'] == 'Strong thunderstorms'],
              df['TotalSpent_RM'][df['weather'] == 'More clouds than sun'],
              df['TotalSpent_RM'][df['weather'] == 'Fog'],
              df['TotalSpent_RM'][df['weather'] == 'Rain'],
              df['TotalSpent_RM'][df['weather'] == 'Overcast'],
              df['TotalSpent_RM'][df['weather'] == 'Mostly cloudy'],
              df['TotalSpent_RM'][df['weather'] == 'Sprinkles'],
              df['TotalSpent_RM'][df['weather'] == 'Haze'],
              df['TotalSpent_RM'][df['weather'] == 'Rain showers']))
    st.write("hypotheses:")
    st.write("• H0: μ1 = μ2 = μ3 = μ4 = μn(all the population means are equal). ")
    st.write("• H1: at least one population mean is different from the rest.")
    st.write("The F-value is apprixmately 1 and the corresponding p-value is 0.462015. F-value = 1 means that no matter what significance level we use for the test, we will conclude that the two variances are equal.")
    st.write("Since the p-value is more than .05, we accept the null hypothesis. Therefore, indicating that we do have sufficient evidence to say that there is no significant difference in weather and total sales.")
    st.write("Q5: Do customers usually visit laundry shop at night?")
    fig=plt.figure(figsize = (10, 7))
    ax=df['Period'].value_counts().plot(kind='bar')
    figs.append(fig)
    st.pyplot(fig)
    st.write("Yes, night period has the most number of customers visiting as compared to other period.")
    st.write("Q6: Do Num_of_Baskets & Basket_Size influence the TotalSpent_RM?")
    model = ols('TotalSpent_RM ~ C(Num_of_Baskets) + C(Basket_Size) + C(Num_of_Baskets):C(Basket_Size)', data=df).fit()
    st.write(sm.stats.anova_lm(model, typ=2))
    st.write("All p-values are more than 0.05, this means that both factors have no statistically significant effect on TotalSpent_RM. The interaction between Num_of_Baskets and Basket_Size does not affect TotalSpent_RM.")
    st.write("Q7: Do Race & TimeSpent_Minutes influence the TotalSpent_RM?")
    model = ols('TotalSpent_RM ~ C(Race) + C(TimeSpent_minutes) + C(Race):C(TimeSpent_minutes)', data=df).fit()
    st.write(sm.stats.anova_lm(model, typ=2))
    st.write("P-values for Race and TimeSpent_Minutes are less than 0.05, this means that both factors have statistically significant effect on TotalSpent_RM. However, the interaction between Race and TimeSpent_Minutes does not affect TotalSpent_RM because the p-value is higher than 0.05.")
    st.write("Q8: Which Race has the highest TotalSpent_RM?")
    fig=plt.figure(figsize = (10, 7))
    ax = sns.boxplot(data = df, x = 'Race', y = 'TotalSpent_RM')
    ax.tick_params(labelrotation=90)
    figs.append(fig)
    st.pyplot(fig)
    st.write("From the comparative boxplots, the medians of Malay, Indian and Chinese are the same. The median of Foreginer is the highest.")
    st.write("Q9: Do having kids affect  TotalSpent_RM, buyDrinks and TimeSpent_minutes?")
    fig=plt.figure(figsize = (10, 7))
    ax = sns.boxplot(data = df, x = 'With_Kids', y = 'TotalSpent_RM')
    ax.tick_params(labelrotation=90)
    fig=plt.figure(figsize = (10, 7))
    ax = sns.boxplot(data = df, x = 'With_Kids', y = 'buyDrinks')
    ax.tick_params(labelrotation=90)
    figs.append(fig)
    st.pyplot(fig)
    fig=plt.figure(figsize = (10, 7))
    ax = sns.boxplot(data = df, x = 'With_Kids', y = 'TimeSpent_minutes')
    ax.tick_params(labelrotation=90)
    figs.append(fig)
    st.pyplot(fig)
    st.write('There is no difference in distribution of datas from the boxplot above thus kids does not affect')

    export_as_pdf = st.button("Export Report")
    if export_as_pdf:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', '', 14)
        pdf.write(5,"Exploratory Data Analysis \n")
        pdf.write(5,"We first add extra data points or features into the table \n")
        pdf.write(5,"Datapoints added are \n")
        pdf.write(5,"1. weather \n")
        pdf.write(5,"2. day of the week \n")
        pdf.write(5,"3. period of the day \n")
        pdf.write(5,"4. is the date a holiday \n")
        pdf.write(5,"5. City of the location \n")
        pdf.write(5,"We replace the city information for some rows that have incorrect value \n")
        pdf.write(5,"Univariate Analysis - Numerical \n")
        texttowrite = [
            "\nAge_Range",
            "\nTimeSpent_minutes",
            "\nbuyDrinks",
            "\nTotalSpent_RM",
            "\nNum_of_Baskets",
            "\nBar plot for all categorical variables in the dataset",
            "\nQ1: Are there more customers visiting during weekends as compared to weekdays?\n   Sunday and Saturday definitely have more customers visiting.\n   Q2: Is there any relationship between the numerical features?",
            "\nCorrelation Between Variables",
            "\nQ3: Is there any relationship between basket size and race?",
            "\nQ4: Did weather information impact the sales?",
            "\nQ5: Do customers usually visit laundry shop at night?",
            "\nQ8: Which Race has the highest TotalSpent_RM?",
            "\nQ9: Do having kids affect TotalSpent_RM, buyDrinks and TimeSpent_minutes?",
            "\nImage2"
        ]
        x=0
        for fig in figs:
            pdf.add_page()
            pdf.write(5,texttowrite[x])
            with NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name)
                    pdf.image(tmpfile.name, 10, 10, 200, 250)
                    x=x+1
        html = create_download_link(pdf.output(dest="S").encode("latin-1"), "testfile")
        st.markdown(html, unsafe_allow_html=True)