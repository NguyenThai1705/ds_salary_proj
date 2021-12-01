# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

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

#train test split
from sklearn.model_selection import train_test_split

X = df_ohe.drop('avg_salary', axis=1)
y = df_ohe.avg_salary.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#multiple linear regression 
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y, X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(X_train, y_train)

np.mean(cross_val_score(lm, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#lasso regression
lm_l = Lasso()

alpha = []
error = []
for i in range(1, 100):
    alpha.append(i/100)
    lm_l = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3)))
    
plt.plot(alpha, error)

err = tuple(zip(alpha, error))
df_err = pd.DataFrame(err, columns=['alpha', 'error'])
df_err[df_err.error == max(df_err.error)]

lm_l = Lasso(alpha=0.07)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#tune models GridsearchCV
from sklearn.model_selection import GridSearchCV

rf_parameters = {'n_estimators': range(10,300,10), 
                 'criterion':('mse', 'mae'), 
                 'max_features':('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, param_grid=rf_parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)

gs.best_estimator_

#test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_gs = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, tpred_lm))
print(mean_absolute_error(y_test, tpred_lml))
print(mean_absolute_error(y_test, tpred_gs))

#saved model
import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ))


#test model
file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(X_test.iloc[1,:].values.reshape(1, -1))
list(X_test.iloc[1,:].values)


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

# predict on df_ohe_new
predict = model.predict(df_ohe_new.values)
predict
