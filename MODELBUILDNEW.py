#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:02:50 2024

@author: palakbajpayee
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv('./Electric_Production (1).csv')  # Replace with your actual file path

# Convert DATE to DATEtime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Feature engineering: Extract useful time-based features from the DATE
data['year'] = data['DATE'].dt.year
data['month'] = data['DATE'].dt.month
data['day'] = data['DATE'].dt.day
data['dayofweek'] = data['DATE'].dt.dayofweek

# Define feature columns and target variable
features = ['year', 'month', 'day', 'dayofweek']
target = 'UNIT'

X = data[features]
y = data[target]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a preprocessor for the numeric and categorical features
numeric_features = ['year', 'month', 'day', 'dayofweek']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Create a pipeline that first preprocesses the data, then fits the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate MAPE
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape:.2%}')



