import streamlit as st
from multiapp import MultiApp
from apps import exploratory,regression # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Exploratory Data Analysis", exploratory.app)
app.add_app("Regression Models", regression.app)
# The main app
app.run()