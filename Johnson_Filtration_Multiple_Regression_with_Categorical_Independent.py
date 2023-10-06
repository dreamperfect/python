# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 11:04:16 2023

@author: 16692
"""

import pandas as pd


import numpy as np
from matplotlib import pyplot as plt 

from statsmodels.formula.api import ols
import seaborn as sns


johnson_filtration=pd.read_excel("C:\\Users\\16692\\Documents\\194 a fall 2021\(5) R files\\Johnson Filtration Multiple Regression with Categorical Independent Variables.xlsx")

johnson_filtration

johnson_filtration.dtypes()



#Change repair type to binary. Select and run def together 
def change_type_of_repair(df):
    df["Type of Repair"] = df["Type of Repair"].replace("electrical", 1)
    df["Type of Repair"] = df["Type of Repair"].replace("mechanical", 0)
    return df

johnson_filtration = change_type_of_repair(johnson_filtration.copy())


johnson_filtration.dtypes


#If you want to make type of repair categorical data type for a legend or something: 
#johnson_filtration["Type of Repair"] = pd.Categorical(johnson_filtration["Type of Repair"])



johnson_filtration

#Allow multiple columns to be displayed
pd.set_option('display.max_columns',None)

johnson_filtration = johnson_filtration.rename(columns={
    'Repair Time in Hours': 'Repair_Time',
    'Months Since Last Service': 'Months_Since_Last_Service',
    'Type of Repair':'Type_of_Repair'
})



johnson_filtration

model=ols("Repair_Time ~ Months_Since_Last_Service + Type_of_Repair", data=johnson_filtration).fit()

model.summary()


predictions=model.predict()

johnson_filtration['Predictions']=predictions


#Anova table 

anova_table = pd.DataFrame({'df': [model.df_model, model.df_resid, model.df_model + model.df_resid],
                            'Sum Sq': [model.ess, model.ssr, model.centered_tss],
                            'Mean Sq': [model.mse_model, model.mse_resid, None],
                            'F value': [model.fvalue, None, None],
                            'Pr(>F)': [model.f_pvalue, None, None]})

anova_table

# Set the index labels
anova_table.index = ['Regression', 'Residual Error', 'Total']

anova_table


#Make a prediction line for electrical and mechanical equation 

tablename=pd.DataFrame()

tablename['Months_Since_Last_Service']=np.linspace(0,10,11)

tablename['Electrical_Predictions']=model.params.Intercept+model.params.Months_Since_Last_Service*tablename.Months_Since_Last_Service+model.params.Type_of_Repair

tablename['Mechanical_Predictions']=model.params.Intercept+model.params.Months_Since_Last_Service*tablename.Months_Since_Last_Service


tablename


#Scatter Plot with Line plots
plt.scatter(johnson_filtration['Months_Since_Last_Service'],johnson_filtration['Repair_Time'],color='black',alpha=1,label="Observed Repair Times")
plt.xlabel('Months_Since_Last Service')
plt.ylabel('Repair_Time')
plt.plot(tablename.Months_Since_Last_Service,tablename.Electrical_Predictions,label="Predicted Electrical Repair Time")
plt.plot(tablename.Months_Since_Last_Service,tablename.Mechanical_Predictions,label="Predicted Mechanical Repair Time")
plt.legend()
plt.show()

johnson_filtration.head()




#Remove Service Call column

johnson_filtration = johnson_filtration.drop(columns=['Service Call'])

johnson_filtration

#Multiple Scatter Plots: 
    
variables=['Months_Since_Last_Service','Type_of_Repair','Repair_Time']

pd.plotting.scatter_matrix(johnson_filtration[variables],figsize=(10,10),alpha=1,color="black")
plt.show()

#You can take out Type of Repair from variables if you want

#Correlation matrix

correlation_matrix=johnson_filtration.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

#Table with predictions, residuals

predictiontable=pd.DataFrame()

predictiontable['Months Since Last Service']=johnson_filtration.Months_Since_Last_Service
predictiontable['Type of Repair']=johnson_filtration.Type_of_Repair
predictiontable['Observed Repair Time']=johnson_filtration.Repair_Time

predictiontable['Predicted Values']=model.predict()
predictiontable['Residuals']=model.resid


#Output table with Mechanical Repairs on top (sort by type of repair ascending)
predictiontable.sort_values('Type of Repair')

predictiontable


#Do the cooks distance, studentized deleted residuals, and leverage/influential observations from ch 15 butler trucking
#code to check for outliers, etc. 
#Also do the normal probability plot if you want to.




