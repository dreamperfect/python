# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:48:58 2024

@author: 16692
"""

#import sktime

from sktime import datasets

 
airline=datasets.load_airline()

airline

print(airline.dtype)

type(airline)


df=airline.to_frame()

df


df.columns=['Passengers']

df

print(airline.index.dtype)


#get a sample of the first 80 percent of the dataset
train=airline.iloc[:-int(len(airline)*0.2)]

train

len(airline)


len(train)

df['naive']=df['Passengers'].shift(1)

df

print(df.index.dtype)

df
#However this does not extend to a forecast
import pandas as pd



#With a period index type of dayes
extra_periods = 12  # for 12 months
new_index = pd.period_range(df.index[-1] + 1, periods=extra_periods, freq='M')

new_index


##this is for when it is in datetime index but right now its in period
#extra_periods = 12  

#new_index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=extra_periods, freq='M')


#if its not in your index put set index

# df=df.set_index('columnname)

#or maybe instead of df.index[-1] do monthlysum['Date of Admission'].iloc[-1]
#new_index = pd.date_range(monthlysum['Date of Admission'].iloc[-1] + pd.DateOffset(months=1), periods=extra_periods, freq='M')
#view file "health care data analysis python" under #do forecasting I suppose to view the rest of how to extend date range by a year without it being 
#in column index.s

df.index

new_index

# Step 2: Create a new DataFrame for the extended periods with NaN values but same column names as df
df_future = pd.DataFrame(index=new_index, columns=df.columns)

df_future


# Step 3: Append the future DataFrame to the original DataFrame
df_extended = pd.concat([df, df_future])

df_extended


# Step 4: Apply the naive forecast using shift (1 month for example)
df_extended['naive'] = df_extended['Passengers'].shift(1)

#click on variable explorer to see full data
df_extended

# Display the extended DataFrame
print(df_extended)


# Plot with naive method (does not require you to change index to timestamp), but index must be the date column

from sktime.utils.plotting import plot_series

plot_series(df_extended['Passengers'], df_extended['naive'], labels = ["Originals", "Naive"])



#second plot method (must convert index to timestamp in order to be able to plot it)

# Converting just df_extended index to timestamp

df_extended.index = df_extended.index.to_timestamp()

df_extended.index.dtype

from matplotlib import pyplot as plt


plt.plot(df_extended.index, df_extended.Passengers,label="Monthly Passengers")

plt.plot(df_extended.index,df_extended.naive, label= "Naive Forecast")

plt.legend()

plt.show()


#A seasonal naive forecast would be taking the data from the previous year same month as a forecast

df_extended['Previous_Years_Month'] = df_extended['Passengers'].shift(12)

df_extended

plt.plot(df_extended.index, df_extended['Passengers'],label="Monthly Passengers")

plt.plot(df_extended.index,df_extended['Previous_Years_Month'], label= "Previous Years Month as Forecast")

plt.legend()

plt.show()



## 3 Month MOVING AVERAGE

df_extended



# Step 1: Calculate the 3-month moving average of the 'Passengers' column (still need to shift down for forecast)

df_extended['Three_monthMA']=df_extended['Passengers'].rolling(window=3).mean


# Step 2: Shift the moving average forward by 1 period to make it a forecast
df_extended['Three_monthMA'] = df_extended['Three_monthMA'].shift(1)


#or just do it in one line of code: 

df_extended['Three_monthMA']=df_extended['Passengers'].shift(1).rolling(window=3).mean

df_extended

#Visualize
plt.plot(df_extended.index, df_extended.Passengers,label="Monthly Passengers")

plt.plot(df_extended.index,df_extended['Three_monthMA'], label= "3 Month Moving Average Forecast")

plt.legend()

plt.show()

df_extended

#Yearly MOVING AVERAGE

df_extended['YearlyMA']=df_extended['Passengers'].rolling(window=12).mean()

# Step 2: Shift the moving average forward by 1 period to make it a forecast
df_extended['YearlyMA'] = df_extended['YearlyMA'].shift(1)

df_extended
#Visualize

plt.plot(df_extended.index, df_extended.Passengers,label="Monthly Passengers")
plt.plot(df_extended.index,df_extended['Three_monthMA'], label= "3 Month Moving Average Forecast")
plt.plot(df_extended.index, df_extended['YearlyMA'], label="Yearly M.A. Forecast")
plt.legend()

plt.show()


#To get R squared (how effective the model was) you first have to drop missing values
#getting R squared for naive forecast

from sklearn.metrics import r2_score

df_extended

#the dropna subset makes sure only the data rows without Na in the columns passengers and naive 
#are kept and therefore removes the extra month prediction in naive forecast since that would be na in the passengers column
# This is important in the R square calculation since you cant compare the prediction to an na in the passengers column. 
#So it drops the rows with index time 1949-01-01 since that is empty in the naive forecast, and it drops after 1960-12-01 
#since after that is a forecast I should probably change the timestamp later to the end of the month maybe since these are months totals.

valid_data = df_extended.dropna(subset=['Passengers', 'naive'])


valid_data[['Passengers','naive']]
    
# Step 2: Get actual values and forecasted values

actual = valid_data['Passengers']
forecast = valid_data['naive']


r_squared = r2_score(actual, forecast)

r_squared

print(round(r_squared,6))

# Adjusted R²
n = len(actual)
p = 0  # number of predictors in naive forecast
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("R²:", round(r_squared, 6))
print("Adjusted R²:", round(adj_r_squared, 6))


#Manual Calculation of r square

actual = valid_data['Passengers']
forecast = valid_data['naive']


import numpy as np


# Calculate the mean of actual values
mean_actual = np.mean(df_extended['Passengers'])

mean_actual


#pd.set_option('display.max_rows', 155)

#print(valid_data['Passengers'])

# Calculate the total sum of squares (SS_tot)
ss_tot = np.sum((actual - mean_actual) ** 2)

ss_tot

# Calculate the residual sum of squares (SS_res)
ss_res = np.sum((actual - forecast) ** 2)

ss_res

# Calculate R-squared
r_squared_manual = 1 - (ss_res / ss_tot)

r_squared_manual

###trend only


from statsmodels.formula.api import ols
import pandas as pd

#repeat the loading in the data set to get a clean dataframe without naive and moving averages

airline=datasets.load_airline()

airline

df=airline.to_frame()


df.columns=['Passengers']
df
#Make a column t which is just the number of the row 1,2,3,4...etc.

df['t'] = range(1, len(df) + 1)

df

model=ols("Passengers ~ t", data=df).fit()

model.summary()

predictions=model.predict()

df['trend only Predictions']=predictions

df





########## Seasonality only

df.index.dtype

#

df['month'] = df.index.month

#if it was not in index as a period, but as a datetime64, you would use this code:
#df_extended['month'] = df_extended['Date of Admission'].dt.month


month_dummies = pd.get_dummies(df['month'], prefix='month')
month_dummies = month_dummies.drop(columns='month_12')  # drop December manually
month_dummies = month_dummies.astype(int)  # convert True/False → 1/0
df = pd.concat([df, month_dummies], axis=1)


df.columns

df

seasonalmodel=ols('Passengers ~ month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',data=df).fit()

seasonalmodel.summary()

predictions=seasonalmodel.predict()

df['seasonality only Predictions']=predictions

df

df.columns

print(seasonalmodel.params)


seasonalmodel.params['Intercept']
seasonalmodel.params

#manually calculating predictions
df['manual_pred'] = seasonalmodel.params['Intercept']

# Loop through each month dummy and add its weighted value
for col in df.columns:
    if col.startswith('month_'):
        df['manual_pred'] += df[col] * seasonalmodel.params[col]


df


############ forecast to the extra periods

extra_periods = 12  # for 12 months
new_index = pd.period_range(df.index[-1] + 1, periods=extra_periods, freq='M')

new_index


##this is for when it is in datetime index but right now its in period
#extra_periods = 12  

#new_index = pd.date_range(df.index[-1] + pd.DateOffset(months=1), periods=extra_periods, freq='M')


df.index

new_index

# Step 2: Create a new DataFrame for the extended periods with NaN values but same column names as df
df_future = pd.DataFrame(index=new_index, columns=df.columns)

df_future

# Fill month column for future data
df_future['month'] = df_future.index.month

df_future[['Passengers','month']]


# --- 3. Create month dummy variables for future months ---
for i in range(1, 12):  # months 1–11 (since month_12 is baseline)
    df_future[f'month_{i}'] = (df_future['month'] == i).astype(int)

# --- 4. Combine original + future data ---
df_extended = pd.concat([df, df_future])

# --- 5. Compute manual predictions for all available rows ---
df_extended['manual_pred'] = seasonalmodel.params['Intercept']

for col in seasonalmodel.params.index:
    if col.startswith('month_'):
        df_extended['manual_pred'] += df_extended[col].fillna(0) * seasonalmodel.params[col]
        
df_extended



#####Manually doing seasonality and trend (must have completed the seasonality only and trend only) to have "t" columns and dummy variable columns

df.columns

seasonalandtrendmodel=ols('Passengers ~ t+ month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',data=df).fit()

seasonalandtrendmodel.summary()

predictions=seasonalandtrendmodel.predict()

df['seasonality & trend Predictions']=predictions

df.columns
#drop the extra columns

df=df[['Passengers','t','seasonality & trend Predictions' ]]

df



#Visualize dataset (must first turn index to timestamp)

df.index = df.index.to_timestamp()


from matplotlib import pyplot as plt


plt.plot(df.index, df['Passengers'],label="Monthly Passengers")


plt.show()


df


###seasonality and trend  


from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score

#This methods only drawback is that you wont get a predicted value for the first entry or row
# SARIMA(p,d,q)(P,D,Q,12)
model = SARIMAX(df['Passengers'],
                order=(1,1,1),
                seasonal_order=(1,1,1,12))
fit = model.fit()

forecast = fit.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

print(forecast.predicted_mean)


# --- Coefficients ---
print("\nModel coefficients:")
print(fit.params)

# --- Predictions (in-sample) ---
pred = fit.fittedvalues
pred

# --- R² and Adjusted R² ---
r2 = r2_score(df['Passengers'], pred)

n = len(df)             # number of observations
p = fit.df_model        # number of estimated parameters
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\nR²:", round(r2, 6))
print("Adjusted R²:", round(adj_r2, 6))

# --- Residuals (error terms) ---
residuals = fit.resid

residuals


# Forecast 12 months ahead
forecast_obj = fit.get_forecast(steps=12)

# Predicted mean (point forecasts)
forecast_mean = forecast_obj.predicted_mean

# Confidence intervals
forecast_ci = forecast_obj.conf_int()

print("Forecast:")
print(forecast_mean)

####visualizing data (first drop the zero value)

# Option 1: Drop first row explicitly
pred = pred.iloc[1:]

pred
# Option 2: Drop where values are zero (if you only want to remove that artificial 0)
pred = pred[pred != 0]


plt.plot(df.index, df['Passengers'],label="Monthly Passengers")

plt.plot(pred.index,pred,label="Prediction Model")

plt.plot(forecast_mean.index, forecast_mean,label="Forecast")

plt.legend()
plt.show()

###Doing 




