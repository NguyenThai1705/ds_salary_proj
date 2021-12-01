# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 10:56:51 2021

@author: nguye
"""
import pandas as pd

df = pd.read_csv('data_eda.csv')

#choose relevant columns
df.columns
df.shape
df_model = df[['Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'comp_count', 'per_hour',
            'employer_provided', 'age', 'python', 'r_studio', 'spark', 'aws', 'excel', 'title_simp', 'seniority',
            'same_location', 'job_state', 'avg_salary']]

#get OneHotEncoder data
# Create a categorical boolean mask
categorical_feature_mask = df_model.dtypes == object
# Filter out the categorical columns into a list for easy reference later on in case you have more than a couple categorical columns
categorical_cols = df_model.columns[categorical_feature_mask].tolist()

# Instantiate the OneHotEncoder Object
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)
# Apply ohe on data
ohe.fit(df_model[categorical_cols])
cat_ohe = ohe.transform(df_model[categorical_cols])

#Create a Pandas DataFrame of the hot encoded column
ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names(input_features = categorical_cols))
#concat with original data and drop original columns
df_ohe = pd.concat([df_model, ohe_df], axis=1).drop(columns = categorical_cols, axis=1)

#how to convert new data to OneHotEncoder Data
columns_model = df_model.columns.drop('avg_salary')
newdata = [[3.6,'1001 to 5000 employees','Company - Private','Gambling',
            'Arts, Entertainment & Recreation','$100 to $500 million (USD)',0,0,0,35,0,0,0,0,1,'data analyst','jr',1,'CA']]

newdf = pd.DataFrame(newdata, columns=columns_model)

# Apply ohe on newdf
cat_ohe_new = ohe.transform(newdf[categorical_cols])
#Create a Pandas DataFrame of the hot encoded column
ohe_df_new = pd.DataFrame(cat_ohe_new, columns = ohe.get_feature_names(input_features = categorical_cols))
#concat with original data and drop original columns
df_ohe_new = pd.concat([newdf, ohe_df_new], axis=1).drop(columns = categorical_cols, axis=1)

data_in = df_ohe_new.values.tolist()
