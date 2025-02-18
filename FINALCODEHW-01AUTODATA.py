# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 12:40:18 2024

@author: sesha
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from matplotlib.pyplot import subplots
#pip install ISLP
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,
summarize,
poly)

auto = pd.read_csv("auto-mpg.csv")
auto = auto.dropna(subset=['horsepower', 'mpg'])
auto.head
auto.columns


#inf values logic
auto.replace([np.inf, -np.inf], np.nan, inplace=True)
auto.dropna(subset=['horsepower', 'mpg'], inplace=True)

#X and y
X = pd.DataFrame({'intercept': np.ones(len(auto)), 'horsepower': auto['horsepower']})
y = auto['mpg']

#convert values to numerics
X['horsepower'] = pd.to_numeric(X['horsepower'], errors='coerce')

#drop nan
combined = pd.concat([X, y], axis=1).dropna()
X = combined[['intercept', 'horsepower']]
y = combined['mpg']

#check dtypes
print(X.dtypes)
print(y.dtypes)

#fit model
model = sm.OLS(y, X).fit()
print(model.summary())
summarize(model)

#preds for 98 HP
preds_df = pd.DataFrame({'intercept': [1], 'horsepower': [98]})
preds_df.head
new_predictions = model.get_prediction(preds_df)
predicted_means = new_predictions.predicted_mean
pred_95 = new_predictions.summary_frame(alpha=0.05)

print(predicted_means)#[24.46707715]
print(pred_95)# MSE 0.251262 


#Plot the response and the predictor. Show the least squares regression line on your plot.
plt.scatter(X['horsepower'], y, label='Data Points')


hp_values = np.linspace(X['horsepower'].min(), X['horsepower'].max(), 100)
mpg_predicted = model.predict(pd.DataFrame({'intercept': np.ones(len(hp_values)), 'horsepower': hp_values}))

plt.plot(hp_values, mpg_predicted, color='red', label='Regression Line')

plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('MPG vs Horsepower with Regression Line')
plt.legend()


plt.show()

###residual plots
# Residuals
residuals = model.resid

# Fitted values
fitted = model.fittedvalues

plt.subplot(2, 2, 1)
plt.scatter(fitted, residuals)
plt.axhline(y=0, color='gray', linestyle='dashed')
plt.xlabel('fitted values')
plt.ylabel('residual')
plt.title('residuals vs fitted')








