# Singapore-Resale-Flat-Prices-Predicting
**Introduction :**

This project aims to develop a machine learning model and create a user-friendly online application to predict accurate resale values for apartments in Singapore. Using historical resale transactions data, the model considers factors such as location, apartment type, size, and lease duration to estimate the future resale price. The goal is to assist both buyers and sellers by providing a reliable prediction tool for evaluating the worth of a previously resold flat. This predictive model addresses challenges associated with diverse influencing factors and empowers users with expected resale prices, enhancing decision-making in the real estate market.

**Domain :**

Real Estate

**Requirements :**

Python
Pandas
Numpy
Streamlit

**Data Source**
Link : https://beta.data.gov.sg/collections/189/view

**Project Workflow**

The following is a fundamental outline of the project:

The Resale Flat Prices dataset has five distinct CSV files, each representing a specific time period. These time periods are 1990 to 1999, 2000 to 2012, 2012 to 2014, 2015 to 2016, and 2017 onwards. Therefore, it is essential to merge the five distinct CSV files into a unified dataset.

The data will be converted into a format that is appropriate for analysis, and any required cleaning and pre-processing procedures will be carried out. Relevant features from the dataset, including town, flat type, storey range, floor area, flat model, and lease commence date will be extracted. Any additional features that may enhance prediction accuracy will also be created.

The objective of this study is to construct a machine learning regression model that utilizes the decision tree regressor to accurately forecast the continuous variable 'resale_price'.

The objective is to develop a Streamlit webpage that enables users to input values for each column and get the expected resale_price value for the flats in Singapore.

**Output**
http://localhost:8501/

**Resale Price Predicting**

To predict the resale price of a Singapore Flats, follow these steps:

Select the "Predictions" option menu.

Fill in the following required information:

Street Name
Block Number
Floor Area (Per Square Meter)
Lease Commence Date
Storey Range
Click the "PREDICT RESALE PRICE" button.

The app will display the predicted resale price based on the provided information.
