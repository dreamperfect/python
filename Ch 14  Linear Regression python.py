# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 18:32:11 2023

@author: 16692
"""

#pg 94 of python notes



# 1st way of doing linear regression ( scipy.stats )


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt 


armand = pd.read_csv("C:\\Users\\16692\\Documents\\194 a fall 2021\\(5) R files\\Armand_Pizza_sheet1.csv")


print(armand)


armand.describe()


pop=armand['Population']

pop

sales=armand['Sales']

sales


from scipy.stats import linregress



res=linregress(pop, sales)

res




predictions=res.intercept+res.slope*pop

predictions




fx=np.array([pop.min(),pop.max()])

fy=res.intercept+res.slope*fx

fx

fy

#I input that ^ into the scatter plot a line between these points  (2,70) and (26,190)
#This would be the same as inputting a line of: (x (2 to 26 or a.k.a population), y=intercept + slope* (2 to 25) or a.k.a predictions)

#To simulate a different population for example:
#Just remember the range should not be higher than the sample data pop 2 to 26.

xs=np.linspace(2,26,25)
xs


xy=res.intercept+res.slope*xs

xy



print(fx)
print(fy)



#select all of these and run together

plt.plot(pop,sales, 'o',label='Recorded Sales')

plt.plot(fx,fy, '-',color="green",label=' Prediction/Linear Regression Line')

plt.legend()

plt.xlabel('Population')
plt.ylabel('Sales')
plt.title('Scatter Diagram (Thousands)')

plt.show()



#Another way to do the Simple Linear Regression (stastmodels)

import statsmodels.formula.api as smf

import statsmodels.stats.anova as anova

#import statsmodels.stats.diagnostic as sm


results=smf.ols('Sales ~ Population', data= armand).fit()

results.params


results.summary()

#View excel for explanation.


confidence_intervals = results.conf_int()

confidence_intervals





#This anova table works only for linear regression
anova_table = anova.anova_lm(results, typ=1)

print(anova_table)

# Residuals

results.resid


#Textbook standardized residuals
influence = results.get_influence()

standardized_residuals = influence.resid_studentized_internal

standardized_residuals

#Sort them in ascending order
sortedstandardizedres=sorted(standardized_residuals,reverse=False)
print(sortedstandardizedres)



#Get predictions 

predictions=res.intercept+res.slope*pop

print(predictions)

#Or another way is 

predictions = results.predict()



#making a table/dataframe

newdf=pd.DataFrame()

newdf['Population']=pop

newdf['Sales']=sales

newdf['Predictions']=predictions

newdf['Residuals']=results.resid

print(newdf)


#Residual Plot Against X

plt.plot(pop,newdf['Residuals'], 'o')

plt.xlabel("Population")
plt.ylabel("Residuals")
plt.title("Plot of Residuals Against Independent Variable X")
plt.show()


#Residual Plot Against Predicted Sales

plt.plot(predictions,newdf['Residuals'], 'o')

plt.xlabel("Predicted Sales")
plt.ylabel("Residuals")
plt.title("Plot of Residuals Against Predicted Sales ")
plt.show()

#Calculating the standardized residuals the way excel does it

excelstdevofresid=np.std(newdf['Residuals'])

excelstandardizedres= newdf['Residuals']/excelstdevofresid
print(excelstandardizedres)

excelsortedstandardizedres=sorted(excelstandardizedres,reverse=False)
print(excelsortedstandardizedres)


# Get the normal scores of a normal distribution, sample size 10, mean 0, standard deviation 1
# Create an array of 10 standard normal deviates

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

excelsortedstandardizedres
#45 Degree line
x=np.linspace(-1.8,1.8, 30)

y=np.linspace(-1.8,1.8, 30)
x

#Normal Probability plot with Excel calculation of Standardized res

plt.plot(normal_scores, excelsortedstandardizedres,'o')
plt.plot(x,y,'-')
plt.xlabel('Normal Scores')
plt.ylabel('Standardized Residuals')
plt.title('Normal Probability Plot')
plt.show()

#Normal Probability plot with Textbook Calculation of Standardized Res

sortedstandardizedres

plt.plot(normal_scores, sortedstandardizedres, 'o')
plt.plot(x,y,'-')
plt.xlabel('Normal Scores')
plt.ylabel('Standardized Residuals')
plt.title('Normal Probability Plot')
plt.show()

#Do the cooks distance, studentized deleted residuals, leverage/influential observations from ch 15 butler trucking
#code to check for outliers, etc.

