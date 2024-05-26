# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:19:46 2024

@author: PALAK BAJPAYEE
"""



import pandas as pd
import numpy as np 
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler # Standardization
from sklearn.preprocessing import MinMaxScaler# Normalization
from urllib.parse import quote
from sqlalchemy import create_engine
from feature_engine.outliers import Winsorizer 
from scipy.stats import skew, kurtosis 

data = pd.read_csv(r"./Electric_Production (1).csv")
user = 'postgres'  # user name
pw = ''  # password
db = 'dataset'  # database name
engine = create_engine(f"postgresql+psycopg2://{user}:%s@localhost/{db}" % quote(f'{pw}'))

data.to_sql('unit', con=engine, if_exists='replace',
           chunksize=1000, index=False)


# SQL query to fetch data
sql_query = "SELECT * FROM unit"



# Load data into DataFrame
df = pd.read_sql(sql_query, engine)
data.head
data.dtypes

################## Check for count of NA's in each column ##################
data.isna().sum()



######################### Outlier detaction ############################

# Let's find outliers in UNIT
sns.boxplot(data.UNIT)               

# Detection of outliers (find limits for UNIT based on IQR)
IQR = data['UNIT'].quantile(0.75) - data['UNIT'].quantile(0.25)

lower_limit = data['UNIT'].quantile(0.25) - (IQR * 1.5)
upper_limit = data['UNIT'].quantile(0.75) + (IQR * 1.5)

# this is only for UNIT colum 
# 1. Remove (let's trim the dataset)
# 3. Winsorization


# Define the model with IQR method
winsor_iqr = Winsorizer(capping_method = 'iqr', 
                        
                          tail = 'both', 
                          fold = 1.5, 
                          variables = ['UNIT'])

df_s = winsor_iqr.fit_transform(data[['UNIT']])

# Let's see boxplot
sns.boxplot(df_s.UNIT)

################# STANDARDIZATION & NORMALIZATION #############################

data.describe()
scaler = StandardScaler()

# Reshape the data to a 2D array and then fit and transform it
df = scaler.fit_transform(data['UNIT'].values.reshape(-1, 1))

# Convert the result back to a DataFrame if needed
df = pd.DataFrame(df, columns=['UNIT_scaled'])

print(df.head())


####################skewness,kurtosis,variance,mean,median,mode############


skewness = skew(data['UNIT'])
variance = data['UNIT'].var()
kurtosis = kurtosis(data['UNIT'])

print("Skewness:", skewness)
print("Variance:", variance)
print("Kurtosis:", kurtosis)

mean = data['UNIT'].mean()
median = data['UNIT'].median()
mode = data['UNIT'].mode() 

print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)




