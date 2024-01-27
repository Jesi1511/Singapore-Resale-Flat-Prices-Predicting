import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import joblib
import numpy as np
from geopy.distance import geodesic
import statistics
import json
import requests
from streamlit_option_menu import option_menu
from PIL import Image


# Define unique values for select boxes
flat_model_options = ['IMPROVED', 'NEW GENERATION', 'MODEL A', 'STANDARD', 'SIMPLIFIED',
                      'MODEL A-MAISONETTE', 'APARTMENT', 'MAISONETTE', 'TERRACE', '2-ROOM',
                      'IMPROVED-MAISONETTE', 'MULTI GENERATION', 'PREMIUM APARTMENT',
                      'ADJOINED FLAT', 'PREMIUM MAISONETTE', 'MODEL A2', 'DBSS', 'TYPE S1',
                      'TYPE S2', 'PREMIUM APARTMENT LOFT', '3GEN']
flat_type_options = ['1 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', '2 ROOM', 'EXECUTIVE', 'MULTI GENERATION']
town_options = ['ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT TIMAH',
                'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG',
                'JURONG EAST', 'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE',
                'QUEENSTOWN', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS',
                'YISHUN', 'LIM CHU KANG', 'SEMBAWANG', 'BUKIT PANJANG', 'PASIR RIS', 'PUNGGOL']
storey_range_options = ['10 TO 12', '04 TO 06', '07 TO 09', '01 TO 03', '13 TO 15', '19 TO 21',
                        '16 TO 18', '25 TO 27', '22 TO 24', '28 TO 30', '31 TO 33', '40 TO 42',
                        '37 TO 39', '34 TO 36', '46 TO 48', '43 TO 45', '49 TO 51']

# Load the saved model
model_filename = r'C:\Users\jesik\OneDrive\Desktop\Singapore_Flat_Resale_-main\resale_price_prediction_linear.joblib'
pipeline = joblib.load(model_filename)



# Sidebar menu for navigation
with st.sidebar:
    selected = option_menu("Menu", ["Home", "Resale Price"],
                           icons=["house", "cash"],
                           menu_icon="menu-button-wide",
                           default_index=0,
                           styles={"nav-link": {"font-size": "20px", "text-align": "left", "margin": "-2px",
                                               "--hover-color": "green"},
                                   "nav-link-selected": {"background-color": "green"}}
                           )

# Main content based on user selection
if selected == 'Home':
    img_path= r"C:\Users\jesik\OneDrive\Desktop\Singapore_Flat_Resale_-main\img.jpg"
    img = Image.open(img_path)

    # Adjust the size of the image as per your requirement
    img_width = 800
    img = img.resize((img_width, int(img_width / img.width * img.height)))
    st.image(img)
    st.markdown("# :green[Singapore Resale Flat Prices Predicting]")
    col1,col2 = st.columns(2)
    st.markdown("####  :violet[*About*] : The Singaporean resale flat market is characterized by intense competition, making it difficult to precisely gauge the resale value of a flat. Numerous variables, including location, flat type, floor area, and lease duration, influence resale prices. The utilization of a predictive model can effectively address these complexities by offering users an estimated resale price derived from these key factors.")

    with col1:
        col1.markdown("#### :violet[*Overview*] : Build regression model to predict resale price")
        col1.markdown("#### :violet[*Domain*] : Resale Flat Prices")
        col1.markdown("#### :violet[*Technologies used*] : Python, Pandas, Numpy, Matplotlib, Seaborn, Scikit-learn, Streamlit.")

    

elif selected == 'Resale Price':
    
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 81%)]")

    # Create a Streamlit sidebar with input fields
    st.sidebar.title("Flat Details")
    town = st.sidebar.selectbox("Town", options=town_options)
    flat_type = st.sidebar.selectbox("Flat Type", options=flat_type_options)
    flat_model = st.sidebar.selectbox("Flat Model", options=flat_model_options)
    storey_range = st.sidebar.selectbox("Storey Range", options=storey_range_options)
    floor_area_sqm = st.sidebar.number_input("Floor Area (sqm)", min_value=0.0, max_value=500.0, value=100.0)
    current_remaining_lease = st.sidebar.number_input("Current Remaining Lease", min_value=0.0, max_value=99.0, value=20.0)
    year = 2024
    lease_commence_date = current_remaining_lease + year - 99
    years_holding = 99 - current_remaining_lease

    # Create a button to trigger the prediction
    if st.sidebar.button("Predict Resale Price"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'town': [town],
            'flat_type': [flat_type],
            'flat_model': [flat_model],
            'storey_range': [storey_range],
            'floor_area_sqm': [floor_area_sqm],
            'current_remaining_lease': [current_remaining_lease],
            'lease_commence_date': [lease_commence_date],
            'years_holding': [years_holding],
            'remaining_lease': [current_remaining_lease],
            'year': [year]
        })

        # Make a prediction using the model
        prediction = pipeline.predict(input_data)

        # Display the prediction
        st.markdown(f'## :violet[Predicted Resale Price:] <span style="color:green">{prediction[0]:,.0f}</span>',
                    unsafe_allow_html=True)
