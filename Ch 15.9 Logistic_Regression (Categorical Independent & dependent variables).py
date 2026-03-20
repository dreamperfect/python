# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 15:00:59 2023

@author: 16692
"""

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


Sim_df=pd.read_excel("C:\\Users\\16692\\Documents\\194 a fall 2021\\(5) R files\\Simmons_Logistic_Regression(Categorical Independent & Dependent Variables).xlsx")

Sim_df.describe()

# Create a two-way contingency table
contingency_table = pd.crosstab(Sim_df['Coupon'], Sim_df['Card'])
print(contingency_table)

# Step 3: Building the model
# Use statsmodels for logistic regression
X = Sim_df[['Spending', 'Card']]
X


X = sm.add_constant(X)  # Add a constant (intercept) to the model
X


y = Sim_df['Coupon']


Sim_logit = sm.Logit(y, X).fit()

# Display the summary of the logistic regression model
print(Sim_logit.summary())

# Fitted values (predictions)
print(Sim_logit.fittedvalues)

# Step 4: Interpreting the Logistic Regression Equation using odds ratios only
odds_ratios = pd.DataFrame({'Odds Ratio': round(np.exp(Sim_logit.params), 2)})
print(odds_ratios)

# Odds ratios and 95% CI


ci = np.exp(Sim_logit.conf_int())
ci.columns = ['2.5%', '97.5%']


ci['Odds Ratio'] = odds_ratios
print(ci)


# Calculate pseudo R-squared manually: 
##not necessary in python since it is output

ll_null=Sim_logit.llnull

ll_null

#log likelihood
ll_proposed=Sim_logit.llf

ll_proposed



pseudo_r_squared = (ll_null - ll_proposed) / ll_null
print("Pseudo R-squared:", pseudo_r_squared)


# p-value for the model
p_value = 1 - stats.chi2.cdf(2 * (ll_proposed - ll_null), df=len(Sim_logit.params) - 1)
print("P-value:", p_value)

y

# Step 5: Create a data frame of probabilities and actual observed values
X['Probability of Coupon'] = Sim_logit.predict()
X['Observed Coupon'] = y

X
# Sort data by predicted probability
X_sorted = X.sort_values(by='Probability of Coupon', ascending=True)

# Add a new column with the rank of each probability
X_sorted['Rank'] = range(1, len(X_sorted) + 1)


# Display the sorted data
print(X_sorted)

# Step 6: Plotting
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.scatterplot(data=X_sorted, x='Rank', y='Probability of Coupon', hue='Observed Coupon')
plt.xlabel("Index")
plt.ylabel("Predicted Probability")
plt.show()


#Confidence matrix with any prediction greater than 0.5 considered a 1 response
conf_matrix=Sim_logit.pred_table()

print(conf_matrix)


#Confidence matrix with any prediction greater than 0.7considered a 1 response

predictions = Sim_logit.predict()  # X is your feature data

predictions
# Define your custom threshold (e.g., 0.7)
custom_threshold = 0.7

# Apply the custom threshold to classify predictions
classified_predictions = (predictions >= custom_threshold).astype(int)

classified_predictions

observed_response=Sim_df.Coupon

observed_response

from sklearn.metrics import confusion_matrix

# Assuming you have classified_predictions as described earlier
# And y_true is your true labels

conf_matrix70 = confusion_matrix(observed_response, classified_predictions)

conf_matrix70

#Calculate leverage and studentized deleted residuals if you want. 


