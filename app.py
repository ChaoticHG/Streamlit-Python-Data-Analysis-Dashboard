import streamlit as st
from multiapp import MultiApp
from apps import exploratory,regression,classification,arm,clustering # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Exploratory Data Analysis", exploratory.app)
app.add_app("Regression Models", regression.app)
app.add_app("Classification Models", classification.app)
app.add_app("Association Rule Mining", arm.app)
app.add_app("Clustering Technique", clustering.app)
# The main app
app.run()