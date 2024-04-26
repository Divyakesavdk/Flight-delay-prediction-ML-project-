import streamlit as st
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the machine learning model
filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

#Read dataset
flights1 = pd.read_csv('flights1.csv')  

# Define the navigation bar
def navbar():
    st.sidebar.title("Menu")
    page = st.sidebar.radio(" ", ["Home", "Predict", "Plots"])
    return page

# Streamlit app title
st.title("FLIGHT DELAY PREDICTION")

# Get the selected page from the navbar
selected_page = navbar()

# Display the selected page
if selected_page == "Home":
    st.header("WELCOME, HAPPY JOURNEY")
    st.image("flight.jpg", use_column_width=True)

elif selected_page == "Predict":
    st.header("Predict Flight Delay")
    # Input form for user
    st.subheader("Enter Flight Details")
    month = st.number_input("Month of Travel", min_value=1, max_value=12, value=1)
    day = st.number_input("Travel day of Month", min_value=1, max_value=31, value=1)
    schdl_dep = st.number_input("Scheduled Departure", value=0.0)
    dep_delay = st.number_input("Delay in Departure", value=0.0, step=0.01)
    schdl_arriv = st.number_input("Scheduled Arrival time", value=0.0)
    divrtd = st.number_input("Was the flight diverted?", min_value=0, max_value=1, value=0)
    cancld = st.number_input("Was the flight cancelled?", min_value=0, max_value=1, value=0)
    air_sys_delay = st.number_input("Delay due to air system", value=0.0, step=0.01)
    secrty_delay = st.number_input("Delay due to security", value=0.0, step=0.01)
    airline_delay = st.number_input("Delay due to Airline", value=0.0, step=0.01)
    late_air_delay = st.number_input("Delay due to late aircraft", value=0.0, step=0.01)
    wethr_delay = st.number_input("Delay due to weather", value=0.0, step=0.01)

    # Button to make predictions
    if st.button("Predict"):
        try:
            # Make predictions using the loaded model
            prediction = loaded_model.predict([[month, day, schdl_dep, dep_delay, schdl_arriv, divrtd, cancld, air_sys_delay, secrty_delay, airline_delay, late_air_delay, wethr_delay]])
            if prediction[0] == 1:
                result = 'will be'
            else:
                result = 'won\'t get'

            # Display prediction result
            st.success(f"The flight {result} delayed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")

elif selected_page == "Plots":
    st.header("Flight Delay Plots")
    
  

    # Example plot 1: Monthly flight count
    st.subheader("Monthly Flight Count")
    plt.figure(figsize=(16, 6))
    plt.style.use('fivethirtyeight')
    ax = sns.countplot(x='MONTH', data=flights1, palette='dark')
    ax.set_xlabel(xlabel='Month', fontsize=18)
    ax.set_ylabel(ylabel='Total flights monthly', fontsize=18)
    st.pyplot(plt)

    # Example plot 2: Daywise flight count
    st.subheader("Daywise Flight Count")
    plt.figure(figsize=(15, 7))
    plt.style.use('ggplot')
    sns.countplot(x='DAY_OF_WEEK', data=flights1, palette='hls')
    plt.title('Daywise flight count', fontsize=16)
    plt.xlabel('Day of the week', fontsize=16)
    plt.ylabel('Total flights everyday', fontsize=16)
    st.pyplot(plt)

    # Example plot 3: Pairplot
    st.subheader("Pairplot of Numerical Features")
    plt.figure(figsize=(12, 8))
    sns.pairplot(flights1[['SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'SCHEDULED_ARRIVAL', 'AIR_SYSTEM_DELAY']])
    st.pyplot(plt)

    # Example plot 4: Correlation Heatmap
    st.subheader("Correlation Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(flights1[['SCHEDULED_DEPARTURE', 'DEPARTURE_DELAY', 'SCHEDULED_ARRIVAL', 'AIR_SYSTEM_DELAY']].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
