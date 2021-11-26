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
            'descrip_len', 'avg_salary']]

#get dummy data
df_dum = pd.get_dummies(df_model)

#train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum.avg_salary.values

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

lm_l = Lasso(alpha=0.18)
lm_l.fit(X_train, y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=3))

#tune models GridsearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

rf_parameters = {'n_estimators': range(10,300,10), 
                 'criterion':('mse', 'mae'), 
                 'max_features':('auto', 'sqrt', 'log2')}

gs = GridSearchCV(rf, param_grid=rf_parameters, scoring='neg_mean_absolute_error', cv=3)
gs.fit(X_train, y_train)



#test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_gs = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(y_test, tpred_lm))
print(mean_absolute_error(y_test, tpred_lml))
print(mean_absolute_error(y_test, tpred_gs))
