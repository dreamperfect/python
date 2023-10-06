# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 14:09:42 2023

@author: 16692
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 
import seaborn as sns
from statsmodels.formula.api import ols

#import statsmodels.formula.api as smf

import statsmodels.api as sm





butler_trucking = pd.read_excel("C:\\Users\\16692\\Documents\\194 a fall 2021\\(5) R files\\Butler_Trucking_Multiple_Regression.xlsx")

butler_trucking

#rename column
butler_trucking=butler_trucking.rename(columns={'Travel hours':'Travel Hours'})

butler_trucking

butler_trucking.describe()



#Multiple Scatter plots
variables = ['Travel Hours', 'Miles Traveled', 'Number of Deliveries']

pd.plotting.scatter_matrix(butler_trucking[variables], figsize=(10, 10),alpha=1,color="black") 

plt.suptitle("Scatter Matrix")
plt.show()

#Just one scatter plot

plt.scatter(butler_trucking['Miles Traveled'],butler_trucking['Travel Hours'],color='black',alpha=1)
plt.xlabel('Miles Traveled')
plt.ylabel('Travel Hours')
plt.show()



correlation_matrix = butler_trucking.corr()




plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


#Without assignment column


columns=['Miles Traveled','Number of Deliveries','Travel Hours']

butler_trucking=butler_trucking[columns]

butler_trucking


correlation_matrix = butler_trucking.corr()

#You can actually see the matrix when its just 3x3 (no assignment column)
correlation_matrix

#Select and run together
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


#Run together, rename with no spaces so its easier.

butler_trucking = butler_trucking.rename(columns={
    'Miles Traveled': 'Miles_Traveled',
    'Number of Deliveries': 'Number_of_Deliveries',
    'Travel Hours':'Travel_Hours'
})


butler_trucking
#Model

model = ols('Travel_Hours ~ Miles_Traveled + Number_of_Deliveries', data=butler_trucking).fit()


#Intercept (bo), b1, b2
print(model.params)

#Model Summary
model.summary()



#Confidence interval 
confidence_intervals = model.conf_int()



#2.5 %     97.5 %
confidence_intervals

#Prediction intervals of y with x1 and x2 independent variables,  95 percent confidence 
prediction_intervals = model.get_prediction(butler_trucking).conf_int(alpha=0.05)

prediction_intervals=pd.DataFrame(prediction_intervals)

prediction_intervals


butler_trucking_with_predict_intervals=pd.concat([butler_trucking, prediction_intervals], axis=1)

pd.set_option('display.max_columns', None)

butler_trucking_with_predict_intervals['Predictions']=np.round(model.predict(),3)

print(butler_trucking_with_predict_intervals)


butler_trucking


#Get the Anova table from textbook


#Degrees of freedom for numerator
model.df_model

#Degrees of freedom in the denominator
model.df_resid

model.ess





anova_table = pd.DataFrame({'df': [model.df_model, model.df_resid, model.df_model + model.df_resid],
                            'Sum Sq': [model.ess, model.ssr, model.centered_tss],
                            'Mean Sq': [model.mse_model, model.mse_resid, None],
                            'F value': [model.fvalue, None, None],
                            'Pr(>F)': [model.f_pvalue, None, None]})

anova_table

# Set the index labels
anova_table.index = ['Regression', 'Residual Error', 'Total']

anova_table



#Residuals

model.resid

#Predictions
model.predict()


#See all of our x1, x2, observed, predicted, residual values in a table

#making a table/dataframe

butler_trucking

table=pd.DataFrame()

table['Miles Traveled']=butler_trucking['Miles_Traveled']

table['Number of Deliveries']=butler_trucking['Number_of_Deliveries']

table['Travel Hours']=butler_trucking['Travel_Hours']

table['Predictions']=np.round(model.predict(),3)

table['Residuals']=np.round(model.resid,3)


#Textbook Calculation of Std of Residuals 

influence = model.get_influence()

standardized_residuals = influence.resid_studentized_internal

standardized_residuals

table['Stdev of Residuals']=standardized_residuals


table


#Get studentized deleted residuals

stud_res = model.outlier_test()

#display studentized deleted residuals



print(stud_res)

#For the ith observation deleted, the data set is n-1, so in the case the error sum of squares
# has (n-1)-p-1 degress of freedom. So for this butler trucking that would be 6 d.f., and 
#at a 0.05 level of significance the t distrubtion of t0.025 with the d.f. is 2.447.
#SO make sure that the studentized deleted residuals is not less than -2.447 or more than 2.447
#Because if so it is an outlier.

model.params


# Get leverage values
#make sure none is greater than 3(p+1)/n which in this case 3(3)/10 equals 0.9
leverage_values = influence.hat_matrix_diag

print(leverage_values)

## Get Cook's distance values
cooks_distance = influence.cooks_distance[0]  # Extract the first element of the tuple

print(cooks_distance)

# Make sure no cooks distance is greater than 1, or else it is an influential observation.





#Predict values with sheet two test set and our model

#Import it by going to variable explorer, import data, choose csv
#change variable name to the one you want, Next, choose Dataframe, Done.

#Rename Columns to the same as butler_trucking



Butler_Trucking_Sheet2.columns=['Miles_Traveled','Number_of_Deliveries']

Butler_Trucking_Sheet2


predictsheet2=model.predict(Butler_Trucking_Sheet2)

predictsheet2


#Get residual plots 

#residual vs fitted


plt.plot(model.predict(),model.resid,'o')
plt.ylabel('Residuals')
plt.xlabel('Predictions')
plt.title('Residual vs Fitted')
plt.show()

#Normal Q-Q (theoretical quantiles, standardized residuals)

sortedstandardizedres=sorted(standardized_residuals,reverse=False)

sortedstandardizedres


from scipy.stats import norm

def ppoints(n, a):
    try:
        n = np.float(len(n))
    except TypeError:
        n = np.float(n)
    return (np.arange(n) + 1 - a)/(n + 1 - 2*a)

#For pppoints down below, change 10(n) to sample size, leave 3/8

ppoints=ppoints(10,3/8)

ppoints

#Normal Scores
normal_scores= norm.ppf(ppoints)

normal_scores

#45 Degree line
x=np.linspace(-1.8,1.8, 30)

y=np.linspace(-1.8,1.8, 30)
x

#Normal Probability plot with Textbook Calculation of Standardized res

plt.plot(normal_scores, sortedstandardizedres,'o')
plt.plot(x,y,'-')
plt.xlabel('Normal Scores')
plt.ylabel('Standardized Residuals')
plt.title('Normal Probability Plot')
plt.show()

#no standardized residual is less than -2 or more than +2 so no outlier is detected that way.




#Scale location ( fitted values (which is predictions), Square root of standardized residuals 0 to 1.4)

model.predict()

sqrtstandardizedres=np.sqrt(np.abs(standardized_residuals))
sqrtstandardizedres

#Residual vs leverage 

plt.plot(model.predict(),sqrtstandardizedres,'o')
plt.title('Scale-Location')
plt.xlabel('Sqrt(Standardized Residual')
plt.ylabel('Predicted Values')







