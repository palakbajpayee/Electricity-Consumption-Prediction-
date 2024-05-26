#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:08:00 2024

@author: palakbajpayee
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from datetime import datetime

# Load the data and train the model
def load_data():
    data = pd.read_csv('/Users/palakbajpayee/Downloads/Electric_Production (1).csv')  # Replace with your actual file path
    data['DATE'] = pd.to_datetime(data['DATE'])
    data['year'] = data['DATE'].dt.year
    data['month'] = data['DATE'].dt.month
    data['day'] = data['DATE'].dt.day
    data['dayofweek'] = data['DATE'].dt.dayofweek
    return data

def train_model(data):
    features = ['year', 'month', 'day', 'dayofweek']
    target = 'UNIT'
    
    X = data[features]
    y = data[target]
    
    numeric_features = features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])
    
    model.fit(X, y)
    
    return model

# Streamlit app
st.title('Electricity Consumption Prediction')

# Load and train the model without caching
data = load_data()
model = train_model(data)

# Input DATE
input_date = st.date_input('Select a DATE', value=datetime.today())

# Extract DATE features
input_year = input_date.year
input_month = input_date.month
input_day = input_date.day
input_dayofweek = input_date.weekday()

# Create input dataframe
input_data = pd.DataFrame({
    'year': [input_year],
    'month': [input_month],
    'day': [input_day],
    'dayofweek': [input_dayofweek]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Predicted UNIT: {prediction[0]:.2f}')


