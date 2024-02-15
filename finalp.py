from datetime import date
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

#icon = Image.open("C:\Final Project\store.jpg")

# SETTING PAGE CONFIGURATIONS
st.set_page_config(
    page_title="Store_Weekly_Sales_Prediction",
    #page_icon=icon,
    layout="wide",
    initial_sidebar_state="expanded")


st.title(':violet[Store_Weekly_Sales_Prediction]') 

submit_button = None  # Define submit_button outside of the block

with st.sidebar:
    selected = option_menu("Menu", ["Home","Explore","Top Chart"],
                           icons =["house","image","toggles", "bar-chart-line","list-task","at"],
                          default_index=0,
                          orientation="vertical",
                          styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin": "0px", 
                                                "--hover-color": "#FF0000"},
                                   "icon": {"font-size": "40px"},
                                   "container" : {"max-width": "2000px"},
                                   "nav-link-selected": {"background-color": "#D3D3D3"},
                                   "nav": {"background-color": "#D3D3D3"}})
    
# READING THE CLEANED DATAFRAME
df = pd.read_csv('C:\\Users\\sunma\\OneDrive\\Documents\\Vscode1\\weeklysalesfinalp.csv')

# HOME MENU
if selected == "Home":
    
    st.markdown(":black_large_square: **Project Title** : Store_Weekly_Sales_Prediction")

    technologies = "streamlit, Machine Learning"
    st.markdown(f":black_large_square: **Technologies** : {technologies}")

    overview = "Streamlit application that allows users are opening a new Store at a particular location. Now, Given the Store Location, Area, Size and other params. Predict the overall weekly sales of the Store."
    st.markdown(f":black_large_square: **Overview** : {overview}")
    #st.image(Image.open("C:\Final Project\sales.jpeg"),width = 400)

# EXPLORE MENU
if selected == "Explore":      

    with st.form("my_form"):
        col1, col2, col3 = st.columns([0.5,0.5,0.1])
    
        with col1:
            Store = st.text_input(label='**Store(Min:1 & Max:45)**')
            Department = st.text_input(label='**Department(Min:1 & Max:99)**')  
            IsHoliday = st.text_input(label='**IsHoliday(Min:0 & Max:1)**')  
            Temperature = st.text_input(label='**Temperature(Min:-5.00 & Max: 105.00)**')
            CPI = st.text_input(label='**CPI(Min:100.0000 & Max: 250.0000)**')
            Unemployment = st.text_input(label='**Unemployment(Min:1.000 & Max: 20.000)**')
            Type = st.text_input(label='**Type(Min:1 & Max:3)**')

        with col2:    
            Size = st.text_input("**Size (Min:1 & Max:300000)**")
            Day = st.text_input(label='**Day(Min:1 & Max:31)**')
            Month = st.text_input(label='**Month(Min:1 & Max:12)**')
            Year = st.text_input(label='**Year(Min:2010 & Max:2012)**')
            Fuel_Price = st.text_input(label='**Fuel_Price(Min:1.000 & Max:5.000)**')
            Total_MarkDown = st.text_input(label='**Total_MarkDown(Min:0.00 & Max:170000.00)**')
            Expected_Weekly_Sales = st.text_input(label='**Expected_Weekly_Sales(Min:0.1 & Max:1000000)**')
        
        with col3:
            store = int(Store) if Store else None
            department = int(Department) if Department else None
            #isholiday = int(IsHoliday) if IsHoliday else None
            isholiday = float(IsHoliday) if IsHoliday else None
            temperature = float(Temperature) if Temperature else None
            cpi = float(CPI) if CPI else None
            unemp = float(Unemployment) if Unemployment else None
            type = int(Type) if Type else None
            size = float(Size) if Size else None  
            day = int(Day) if Day else None  
            month = int(Month) if Month else None   
            year = int(Year) if Year else None 
            fuel_price = np.log(float(Fuel_Price)) if Fuel_Price else None
            total_MarkDown = float(Total_MarkDown) if Total_MarkDown else None
            expected_weekly_sale = np.log(float(Expected_Weekly_Sales)) if Expected_Weekly_Sales else None
        
        
    # Form submission button
    submit_button = st.form_submit_button("Submit")

    # Load the model and scaler
    with open('C:\\Users\\sunma\\OneDrive\\Documents\\Vscode1\\salesmodel.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    with open('C:\\Users\\sunma\\OneDrive\\Documents\\Vscode1\\salesscaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

def predict_sales(sample):
    # Convert IsHoliday, Type, Day, Month, and Year to categorical
    sample['IsHoliday'] = sample['IsHoliday'].astype(int).astype(str)
    sample['Type'] = sample['Type'].astype(int).astype(str)
    sample['Day'] = sample['Day'].astype(int).astype(str)
    sample['Month'] = sample['Month'].astype(int).astype(str)
    sample['Year'] = sample['Year'].astype(int).astype(str)

    # Apply one-hot encoding to categorical features
    sample_encoded = pd.get_dummies(sample, drop_first=True)
    
    # Ensure all columns are present
    expected_columns = ['Store', 'Department', 'IsHoliday_1', 'Temperature', 'CPI', 'Unemployment', 'Type_2', 'Type_3',
                        'Size', 'Day_2', 'Day_3', 'Day_4', 'Day_5', 'Day_6', 'Day_7', 'Day_8', 'Day_9', 'Day_10',
                        'Day_11', 'Day_12', 'Day_13', 'Day_14', 'Day_15', 'Day_16', 'Day_17', 'Day_18', 'Day_19',
                        'Day_20', 'Day_21', 'Day_22', 'Day_23', 'Day_24', 'Day_25', 'Day_26', 'Day_27', 'Day_28',
                        'Day_29', 'Day_30', 'Day_31', 'Month_2', 'Month_3', 'Month_4', 'Month_5', 'Month_6', 'Month_7',
                        'Month_8', 'Month_9', 'Month_10', 'Month_11', 'Month_12', 'Year_2011', 'Year_2012', 'Fuel_Price',
                        'Total_MarkDown', 'Expected_Weekly_Sales']

    for col in expected_columns:
        if col not in sample_encoded.columns:
            sample_encoded[col] = 0
    
    # Select only the expected columns
    sample_encoded = sample_encoded[expected_columns]

    # Scale the numerical features
    sample_scaled = scaler_loaded.transform(sample_encoded)
    
    # Make prediction
    prediction = loaded_model.predict(sample_scaled)
    
    return np.exp(prediction)[0]

# Use the user inputs for prediction
if submit_button:
    # Create feature array
    Inputs = [store,department, isholiday,temperature,CPI,Unemployment,type, size,year,fuel_price,total_MarkDown,expected_weekly_sale]

    # Make prediction
    prediction_result = predict_sales(np.array(Inputs))
    st.success(f'Predicted Weekly Sales: ${prediction_result:.2f}')
    st.balloons()
