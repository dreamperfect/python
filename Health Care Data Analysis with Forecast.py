# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 16:26:33 2025

@author: 16692
"""

#Link to data set:
#https://www.kaggle.com/datasets/prasad22/healthcare-dataset/data?select=healthcare_dataset.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

healthcaredf = pd.read_excel("C:\\Users\\16692\\Documents\\194 a fall 2021\\(5) R files\\Leetcode practice problems\\Healthcare data set kaggle.xlsx", sheet_name="import this")


#outputs count of duplicate rows.
healthcaredf.duplicated().sum()

#filter for duplicates


healthcaredf[healthcaredf.duplicated(keep=False)].sort_values('Name')

#drop duplicates

healthcaredf=healthcaredf.drop_duplicates()
print(len(healthcaredf))

healthcaredf.duplicated().sum()



import matplotlib.ticker as ticker
pd.set_option('display.max_columns', None)
import numpy as np





#See the column names, datatypes, non null row count and null count. 

healthcaredf.info()

healthcaredf.head()
healthcaredf.head().T

#Theres no null rows, and theres 54,966 rows.


# Check if the billing amount has $ and commas signs in it, and if it has more than 2 decimals
healthcaredf["Billing Amount"]


# Check if any values contain parentheses
has_parentheses = healthcaredf['Billing Amount'].astype(str).str.contains(r'\(.*\)')

# Count how many rows have parentheses
print(has_parentheses.sum())

# Optionally, view only the rows that have parentheses
print(healthcaredf[has_parentheses])


# Remove $ and commas
billing_clean = healthcaredf["Billing Amount"].astype(str).str.replace(r'[\$,]', '', regex=True)


# Handle parentheses for negatives
billing_clean = billing_clean.str.replace(r'\((.*)\)', r'-\1', regex=True)

# Convert to float
healthcaredf["Billing Amount"] = billing_clean.astype(float)


#Make sure there are no rows that failed to convert
healthcaredf["Billing Amount"].isna().sum()


healthcaredf["Billing Amount"].sum()

#Total Billing amount 1,417,432,041.3952546

healthcaredf.dtypes


#check if there are negative numbers in billing amount

negatives=healthcaredf[healthcaredf['Billing Amount']<0]

print(negatives[['Name','Billing Amount']])

#check length of negative billing amounts

len(negatives)


#Fix the random capitalization and lowercase in the names column

healthcaredf['Name'] = healthcaredf['Name'].str.title()


# Convert dates
healthcaredf["Date of Admission"] = pd.to_datetime(healthcaredf["Date of Admission"], errors='coerce')
healthcaredf["Discharge Date"] = pd.to_datetime(healthcaredf["Discharge Date"], errors='coerce')

healthcaredf["Length of Stay (days)"] = (healthcaredf["Discharge Date"] - healthcaredf["Date of Admission"]).dt.days


#Check value counts of object datatypes to decide if it should be a category 
#When there are few unique values they all output or (length doesnt output since it shows all unique))
#Chat gpt says when the number of unique values is like 5 to 10% of total rows maybe you should make it a category
#so with 55,500 rows that is between 2.775 to 5,550 unique values tops but really it says you should for sure if it
#is 1% or less so that is 555 unique values.

categorical_cols=healthcaredf.select_dtypes('object')

for i in categorical_cols:
    print("\n Unique Values ")
    print(healthcaredf[i].value_counts())
    
#So the ones that should be converted to categories are: Gender, Blood Type, Medical Condition, 
   # Insurance Provider, Admission Type, Medication, Test Results


# Convert to category
categorical_cols = [
    "Gender", "Blood Type", "Medical Condition",
    "Insurance Provider", "Admission Type", "Medication", "Test Results"]

healthcaredf[categorical_cols] = healthcaredf[categorical_cols].astype("category")


# Check result
healthcaredf.info()



#Assessing ages 

healthcaredf['Age'].describe()

healthcaredf['Age'].mode()

len(healthcaredf[healthcaredf['Age'] == 38])


#Minimum age is 13, max is 89, most common age is 38 which appeared 897 times.


# Adding a new column that labels the age group of the patient

bn=[10,18,25,65,90]
label=['Teens(10-17)','Young Adults(18-24)','Adults(25-64)','Seniors(>65)']
healthcaredf['Age Group']=pd.cut(healthcaredf['Age'],bins=bn,labels=label,right=False)

healthcaredf[['Age Group','Age']]

healthcaredf['Age Group'].value_counts()

########Histogram of ages

healthcaredf['Age Group'].hist()
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

#barplot of ages count

age_counts = healthcaredf['Age Group'].value_counts().sort_index()
age_counts

plt.bar(age_counts.index, age_counts.values)
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Number of Patients per Age Group')
plt.xticks(rotation=45)  # rotate labels if needed
for i, v in enumerate(age_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontsize=10)  # adds count above bar
plt.show()


#The percentage of patients makes it show up on the bar plot

age_counts_pct = (age_counts / age_counts.sum()) * 100

plt.bar(age_counts_pct.index, age_counts_pct.values)
plt.ylabel('Percentage of Patients')
plt.title('Percentage of Patients per Age Group')
plt.xticks(rotation=45, ha='right')  # rotate and align labels
plt.show()


#######1 Average billing amount by medical condition#############################


avg_billing_by_condition = (healthcaredf.groupby("Medical Condition")["Billing Amount"].mean().sort_values(ascending=False).round(2))


avg_billing_by_condition


#Bar plot

plt.figure(figsize=(10,5))
ax = sns.barplot(
    x=avg_billing_by_condition.index,
    y=avg_billing_by_condition.values,
    order=avg_billing_by_condition.index,  # ensures correct sort order
    palette="viridis"
)
# Add value labels on top of bars
for i, v in enumerate(avg_billing_by_condition.values):
    ax.text(i, v, f"${v:,.0f}", ha='center', va='bottom', fontsize=9) #adds bar label, with dollar sign and rounds the nearest dollar. 
# Labels and formatting
plt.title("Average Billing Amount by Medical Condition", fontsize=14)
plt.xlabel("Medical Condition")
plt.ylabel("Average Billing ($)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



######### 3 Test whether 3 or more groups (Admission types) have significantly different mean billing amounts ######
from scipy.stats import f_oneway

emergency = healthcaredf.loc[healthcaredf["Admission Type"]=="Emergency", "Billing Amount"]

urgent = healthcaredf.loc[healthcaredf["Admission Type"]=="Urgent", "Billing Amount"]
elective = healthcaredf.loc[healthcaredf["Admission Type"]=="Elective", "Billing Amount"]

f_stat, p_val = f_oneway(emergency, urgent, elective)
print(f"ANOVA p-value: {p_val:.5f}")

#f p_value is less than 0.05: Reject H₀ → there is a statistically significant difference in average billing amounts
# between at least two admission types.
#If p value is more than 0.05, fail to reject Ho, therefore there is no difference in average billing amounts





#####2 Average Length of stay by admission type (barplot)#############################
healthcaredf.info()

# Calculate stay duration (in days)
healthcaredf["Length of Stay (days)"] = (
    (healthcaredf["Discharge Date"] - healthcaredf["Date of Admission"]).dt.days)

healthcaredf['Length of Stay (days)']

avg_stay_by_admission = (
    healthcaredf.groupby("Admission Type")["Length of Stay (days)"].mean().sort_values(ascending=False))

print(avg_stay_by_admission)

plt.figure(figsize=(8,5))
sns.barplot(x=avg_stay_by_admission.index, y=avg_stay_by_admission.values)

for i, v in enumerate(avg_stay_by_admission.values):
    plt.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=9)  # va='bottom' puts it just above the bar

plt.title("Average Length of Stay by Admission Type")
plt.xlabel("Admission Type")
plt.ylabel("Average Stay (days)")
plt.show()

#Checking the unique types of admission types
healthcaredf['Admission Type'].unique()

####histogram of x axis length of stay (Days), y axis (count of times that occurs in dataset ) ############


healthcaredf['Length of Stay (days)'].hist()
plt.xlabel("Days Stayed")
plt.ylabel("Count")
plt.show()





#################2.5 Average length of stay by medical condition  ####################3


avg_stay_by_medicalondition = (
    healthcaredf.groupby("Medical Condition")["Length of Stay (days)"].mean().sort_values(ascending=False)
)


print(avg_stay_by_medicalondition)



######3 which medical conditions most often have abnormal test results (barplot)#######

abnormal_counts = (
    healthcaredf[healthcaredf["Test Results"] == "Abnormal"]
    .groupby("Medical Condition")["Name"].count()
    .sort_values(ascending=False)
)
print(abnormal_counts)

#Set custom colors 
colors = {
    'Arthritis':   '#d62728',      #red
    'Diabetes':'#1f77b4',          # blue
    'Obesity': '#2ca02c',         # Green
    'Cancer':'#9467bd',    # Purple
    'Hypertension':  '#ff7f0e',        # Orange
    'Asthma': '#8c564b'           # Brown
}

#bar plot with set custom colors 

plt.figure(figsize=(10,5))
sns.barplot(x=abnormal_counts.index, y=abnormal_counts.values,order=abnormal_counts.index,
            palette=[colors[cond] for cond in abnormal_counts.index])
for i, v in enumerate(abnormal_counts.values):
    plt.text(i, v, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
plt.title("Abnormal Test Results by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Count of Abnormal Results")
plt.xticks(rotation=45)
plt.show()

#bar plot with palette default argument that is for categorical 


plt.figure(figsize=(10,5))
sns.barplot(x=abnormal_counts.index, y=abnormal_counts.values,order=abnormal_counts.index, palette="Set1")
for i, v in enumerate(abnormal_counts.values):
    plt.text(i, v, f"{v:.0f}", ha='center', va='bottom', fontsize=9)
plt.title("Abnormal Test Results by Medical Condition")
plt.xlabel("Medical Condition")
plt.ylabel("Count of Abnormal Results")
plt.xticks(rotation=45)
plt.show()





####4 Billing amount distribution by insurance provider (boxplot)################

#value counts of insurance provider instances
healthcaredf['Insurance Provider'].value_counts()

#there are no missing values but if there were then I'd use this:
healthcaredf['Insurance Provider'].value_counts(dropna=False)

#shows min, max, median, iqr
plt.figure(figsize=(10,6))
sns.boxplot(data=healthcaredf, x="Insurance Provider", y="Billing Amount")
plt.title("Billing Amount Distribution by Insurance Provider")
plt.xlabel("Insurance Provider")
plt.ylabel("Billing Amount ($)")
plt.xticks(rotation=45)
plt.show()

#Manually seeing average billing amount by insurance provider
agg_df = healthcaredf.groupby("Insurance Provider")["Billing Amount"].mean()
print("Billing Amount by")
print(agg_df)

#####5 relationship between age and billing amount ##############################

plt.figure(figsize=(8,5))
sns.scatterplot(data=healthcaredf, x="Age", y="Billing Amount", hue="Gender", alpha=0.6)
plt.title("Relationship Between Patient Age and Billing Amount")
plt.xlabel("Age")
plt.ylabel("Billing Amount ($)")
plt.show()

#relational plot (just two scatter plots with gender as rows)

sns.relplot(x='Age',y='Billing Amount', data=healthcaredf, kind='scatter',row='Gender',hue='Gender')
plt.show()


# Correlation
correlation = healthcaredf["Age"].corr(healthcaredf["Billing Amount"])
print(f"Correlation between Age and Billing Amount: {correlation:.3f}")

from scipy.stats import pearsonr

r, p_value = pearsonr(healthcaredf["Age"], healthcaredf["Billing Amount"])

print(f"r = {r:.3f}, p = {p_value:.3f}")

#If the p_value > 0.05 then it is not a statistically significant relationship.

###top doctor visits

doctor_counts = healthcaredf['Doctor'].value_counts().nlargest(10).reset_index()

doctor_counts

doctor_counts = healthcaredf['Doctor'].value_counts().head(10).reset_index()

doctor_counts


doctor_counts.columns = ['Doctor', 'Patient Count']

doctor_counts


 
##########   Average billing grouped by hospital stay length   ##############################


# Calculate stay duration (in days) again
healthcaredf["Length of Stay (days)"] = (
    (healthcaredf["Discharge Date"] - healthcaredf["Date of Admission"]).dt.days)

healthcaredf.info()


avg_bill_by_stay_length = (healthcaredf.groupby("Length of Stay (days)")["Billing Amount"].mean()
    .round(2)
    .reset_index()
    .rename(columns={"Billing Amount": "Average Billing Amount"})
)

avg_bill_by_stay_length

#Bar Plot of Average Billing amount by length of stay (the default of the sns.barplot() is average on y 
#   if you dont put estimator=sum )
 
sns.barplot(data=healthcaredf, x="Length of Stay (days)", y="Billing Amount")
plt.title("Billing Amount Average by Length of Stay (DAYS)")
plt.xlabel("Days Stayed in Hospital")
plt.ylabel("Billing Amount ($)")
plt.show()

#Bar Plot of Summed Billing amount by length of stay 


sns.barplot(data=healthcaredf, x="Length of Stay (days)", y="Billing Amount", estimator=sum, ci=None)
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.show()



##########Top ten Hospitals with highest average billing by medical condition


healthcaredf['Medical Condition'].unique()

topbillinghospitalpercondition= (healthcaredf.groupby(['Hospital', 'Medical Condition'])['Billing Amount'].mean()
    .sort_values(ascending=False)
    .head(10).reset_index())

topbillinghospitalpercondition

#Note that this ^ outputs some repeating medical conditions because they are in the top ten highest billing 
 
#What I really want is the Top ten Hospitals with highest average billing, then the bar plot colored in by their average billing per medical condition
#but this wont be possible unless I subset the data for only the hospitals with all 6 medical conditions in the data

#but not all hospitals have 6 medical conditions in the data set

#number of unique hospitals
print(healthcaredf.groupby('Hospital').nunique())
#39,876 hospitals

#Filter for hospitals with all 6 medical conditions treated in their data: 
    
hospitals_num_condition = (healthcaredf.groupby('Hospital')['Medical Condition'].nunique().reset_index())

hospitals_num_condition

pd.set_option('display.max_rows', None)
pd.reset_option('display.max_rows')

hospitals_with_all6 = hospitals_num_condition[hospitals_num_condition['Medical Condition'] == 6]

hospitals_with_all6

mask6=hospitals_with_all6['Hospital']
mask6

#only 106 hospitals have all 6 medical conditions in their data set. but how many rows of data are they in the original/ filter original data for just those

thosewith6=healthcaredf[healthcaredf['Hospital'].isin(mask6)]

thosewith6
#1807 rows 

#Hospitals with at least 3 medical conditions treated in their data

hospitals_num_condition = (healthcaredf.groupby('Hospital')['Medical Condition'].nunique().reset_index())


hospitals_with_3ormore = hospitals_num_condition[hospitals_num_condition['Medical Condition'] >= 3]

mask3=hospitals_with_3ormore['Hospital']

len(mask3)

#1,422 hospitals have data with at least 3. Filter the data to only those hospitals:

thosewith3plus=healthcaredf[healthcaredf['Hospital'].isin(mask3)]

thosewith3plus

#there are 9,030 rows of data for those 


#A. find top 10 hospitals by average billing overall of only the hospitals that have all 6 medical conditions treated
#the .index gets just the names of the hospitals
top10_hospitals = (thosewith6.groupby(['Hospital'])['Billing Amount'].mean()
    .sort_values(ascending=False)
    .head(10).index)

top10_hospitals

#B. Filter data to only those hospitals 
filtered = healthcaredf[healthcaredf['Hospital'].isin(top10_hospitals)]

filtered


# C. Group by Hospital and Medical Condition to get average billing
avg_billing = (
    filtered.groupby(['Hospital', 'Medical Condition'])['Billing Amount']
    .mean()
    .reset_index())

avg_billing

# D. Pivot so Medical Conditions become columns for stacked bars
pivot_df = avg_billing.pivot(index='Hospital', columns='Medical Condition', values='Billing Amount')

pivot_df

# **Sort hospitals by total average billing**
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).index]


pivot_df


# E. Create the stacked bar plot
pivot_df.plot(kind='bar', stacked=True, figsize=(10,6), colormap='tab20')

plt.title('Average Billing by Medical Condition for Top 10 Hospitals')
plt.ylabel('Average Billing Amount ($)')
plt.xlabel('Hospital')
plt.xticks(rotation=45, ha='right')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

plt.legend(title='Medical Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


##########Billing Amount According to Medical Condition and Medication


df_trans = healthcaredf.groupby(['Medical Condition', 'Medication'])[['Billing Amount']].sum().reset_index()

plt.figure(figsize=(15,6))
sns.barplot(x=df_trans['Medical Condition'], y=df_trans['Billing Amount'], hue=df_trans['Medication'], ci=None, palette="Set1")
plt.title("Billing Amount Sum according to Medical Condition and Medication")
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.ylabel("Billing Amount")
plt.xticks(rotation=45, fontsize = 9)
plt.show()

######Billing amount according to medical condition and test results


df_trans = healthcaredf.groupby(['Medical Condition', 'Test Results'])[['Billing Amount']].sum().reset_index()

plt.figure(figsize=(15,6))
sns.barplot(x=df_trans['Medical Condition'], y=df_trans['Billing Amount'], hue=df_trans['Test Results'], ci=None, palette="Set1")
plt.title("Billing Amount according to Medical Condition and Test Results")
plt.ylabel("Billing Amount")
plt.xticks(rotation=45, fontsize = 9)
plt.show()

#### categorical summaries 

healthcaredf.info()

cols = ['Gender','Blood Type', 'Medical Condition',
        'Insurance Provider', 'Admission Type',
        'Medication', 'Test Results','Age Group']

#This code does not need to be an if else since the gender plot is the same 
for i in cols:  
    if i == 'Gender':
        fig, ax = plt.subplots(1, 2) 
        fig.suptitle('** Gender **', fontsize=20) 
        plt.style.use('seaborn')
        plt.subplot(1,2,1)
        healthcaredf['Gender'].value_counts().plot(kind='bar',color=sns.color_palette("tab10"))
        plt.subplot(1,2,2)
        healthcaredf['Gender'].value_counts().plot(kind='pie',autopct="%.2f%%")
        plt.show()
    
    else:
        fig, ax = plt.subplots(1, 2) 
        fig.suptitle('** ' + i + ' **', fontsize=20) 
        plt.style.use('seaborn')
        plt.subplot(1,2,1)
        healthcaredf[i].value_counts().plot(kind='bar',color=sns.color_palette("tab10"))
        plt.subplot(1,2,2)
        healthcaredf[i].value_counts().plot(kind='pie',autopct="%.2f%%")
        plt.show()



############## Ten Hospitals with highest average number of days hospitalized#########

healthcaredf.info()

toplenghthstayhospitals=healthcaredf.groupby('Hospital')['Length of Stay (days)'].mean().sort_values(ascending=False).head(10)

toplenghthstayhospitals




####For loop that iterates on cols. Highest 'Features' according to average number of days hospitalized



healthcaredf.columns

healthcaredf['Days hospitalized']=healthcaredf['Length of Stay (days)']

healthcaredf.info()

cols = ['Hospital','Gender','Blood Type','Medical Condition',
        'Insurance Provider','Admission Type',
        'Medication','Test Results']


import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = 'browser'



for i in cols:
    char_bar = (
        healthcaredf.groupby(i)[['Days hospitalized']]
        .mean()
        .reset_index()
        .sort_values(by="Days hospitalized", ascending=False)
    )

    top = char_bar.head(10)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=top[i], y=top["Days hospitalized"]))

    fig.update_layout(
        title=f'Average number of days hospitalized grouped by {i}',
        xaxis_title=i,
        yaxis_title="Days hospitalized",
        plot_bgcolor='black',
        paper_bgcolor='gray',
        font=dict(color='white')
    )

    fig.show()




######## Reworking Above code to an if else loop to change title for hospital since that one is top 10 

 
for i in cols:
    if i == 'Hospital':
        char_bar = (
            healthcaredf.groupby('Hospital')[['Days hospitalized']]
            .mean()
            .reset_index()
            .sort_values(by="Days hospitalized", ascending=False)
        )

        top = char_bar.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top[i], y=top["Days hospitalized"]))

        fig.update_layout(
            title=f'Top Ten Average Number of Days Hospitalized Grouped by Their {i}s',
            xaxis_title=i,
            yaxis_title="Avg Days hospitalized",
            plot_bgcolor='black',
            paper_bgcolor='gray',
            font=dict(color='white'))
        fig.show() 
    else:
        char_bar = (
            healthcaredf.groupby(i)[['Days hospitalized']]
            .mean()
            .reset_index()
            .sort_values(by="Days hospitalized", ascending=False)
        )

        top = char_bar.head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top[i], y=top["Days hospitalized"]))

        fig.update_layout(
            title=f'Average Number of Days Hospitalized Grouped by {i}',
            xaxis_title=i,
            yaxis_title="Avg Days hospitalized",
            plot_bgcolor='black',
            paper_bgcolor='gray',
            font=dict(color='white'))
        fig.show()
        
### Let me see if I can make it a plotly express

import plotly.express as px

for i in cols:
    if i == 'Hospital':
        char_bar = (
            healthcaredf.groupby('Hospital')[['Days hospitalized']]
            .mean()
            .reset_index()
            .sort_values(by="Days hospitalized", ascending=False)
        )

        top = char_bar.head(10)
        fig = px.bar(data_frame=top, x=top[i],y=top['Days hospitalized'])
        fig.update_layout(
            title=f' Top 10 {i}s by Average Length of Stay',
            xaxis_title=i,
            yaxis_title="Avg. Days hospitalized",
            plot_bgcolor='black',
            paper_bgcolor='gray',
            font=dict(color='white'))
        fig.show() 
    else:
        char_bar = (
            healthcaredf.groupby(i)[['Days hospitalized']]
            .mean()
            .reset_index()
            .sort_values(by="Days hospitalized", ascending=False)
        )

        top = char_bar.head(10)
        fig = px.bar(data_frame=top, x=top[i],y=top['Days hospitalized'])
        fig.update_layout(
            title=f' Average Number of Days Hospitalized Grouped by {i}',
            xaxis_title=i,
            yaxis_title="Avg. Days hospitalized",
            plot_bgcolor='black',
            paper_bgcolor='gray',
            font=dict(color='white'))
        fig.show()
        



#checking medical condition grouped by avg days hospitlized plot

conditions_mean_hospitalization_stay =healthcaredf.groupby('Medical Condition')['Days hospitalized'].mean().reset_index()

conditions_mean_hospitalization_stay.dtypes

type(conditions_mean_hospitalization_stay)


conditions_mean_hospitalization_stay

####### Practicing plotly figures outside of loop ################################


import plotly.graph_objects as go

fig=go.Figure()
fig.add_trace(go.Bar(x=conditions_mean_hospitalization_stay['Medical Condition'],y=conditions_mean_hospitalization_stay['Days hospitalized']))
fig.show()

###############As plotly express ##################################################

import plotly.express as px
conditions_mean_hospitalization_stay


fig=px.bar(data_frame=conditions_mean_hospitalization_stay,x="Medical Condition",y="Days hospitalized")
fig.show()



healthcaredf.info()

healthcaredf.columns

#How many hospitals are in the dataset? 

healthcaredf['Hospital'].value_counts()

#How many hospitals occur more than 10 times in data set (162)

healthcaredf['Hospital'].value_counts()[healthcaredf['Hospital'].value_counts() > 10]


#filter for those hospitals with value counts over ten

over10=healthcaredf[healthcaredf.groupby('Hospital')['Hospital'].transform('count') > 10]


over10['Hospital'].value_counts()

over10['Hospital'].value_counts().index


#get index of name hospitals with over 10 rows (notice it says length 162 so 162 hospitals)

healthcaredf['Hospital'].value_counts()[healthcaredf['Hospital'].value_counts() > 10].index







healthcaredf.head()

print(len(healthcaredf))






###### Sorting most of Entire Dataset by date of admission, then resampling to monthly sum of visits    #############################

timeseries=healthcaredf[['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition',
       'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider',
       'Billing Amount', 'Room Number', 'Admission Type', 'Discharge Date',
       'Medication', 'Test Results', 'Age Group', 'Length of Stay (days)']].sort_values('Date of Admission').reset_index()

timeseries.head()

countofvisitsperday=timeseries['Date of Admission'].value_counts().reset_index().sort_values(by='Date of Admission')

countofvisitsperday


countofvisitsperday.plot(x='Date of Admission',y='count', kind='line')
plt.show()

#too busy so switching to monthly sum (resampling to sum of visits per month)

import matplotlib.dates as mdates

monthlysum= countofvisitsperday.resample('M',on='Date of Admission').sum().reset_index()

monthlysum

monthlysum.dtypes

monthlysum.plot(x='Date of Admission',y='count', kind='line')
plt.xlabel('Date')
plt.ylabel("Number of Patients")
plt.title('Monthly Number of Hospital Patients')
plt.show()


#with year-month=day 
plt.plot(monthlysum['Date of Admission'],monthlysum['count'])
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.show()

monthlysum.tail()

### Use forecasting technique I suppose

monthlysum.dtypes

monthlysum['Date of Admission'].min(), monthlysum['Date of Admission'].max()


monthlysum




####################     With adding the 12 periods at the beginning
extra_periods = 12  
new_dates = pd.date_range(monthlysum['Date of Admission'].iloc[-1] + pd.DateOffset(months=1), periods=extra_periods, freq='M')

new_dates



# Step 2: Create a new DataFrame for the extended periods with NaN values but same column names as df
df_future = pd.DataFrame(columns=monthlysum.columns)

df_future['Date of Admission']=new_dates

df_future


# Step 3: Append the future DataFrame to the original DataFrame
df_extended = pd.concat([monthlysum, df_future],ignore_index=True)

df_extended

df_extended.dtypes

#Naive Method

# Step 4: Apply the naive forecast using shift (1 month for example)
df_extended['naive'] = df_extended['count'].shift(1)

df_extended





#second plot method (must convert index to timestamp in order to be able to plot it)

# Converting just df_extended 'date of admission' to timestamp or maybe dont (fine as datetime)

df_extended.dtypes

#df_extended['Date of Admission'] = df_extended['Date of Admission'].to_timestamp()


plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals")
plt.plot(df_extended['Date of Admission'],df_extended['naive'], label= "Naive Forecast")
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

valid_data = df_extended.dropna(subset=['count', 'naive'])


valid_data[['count','naive']]
    
# Step 2: Get actual values and forecasted values

actual = valid_data['count']
forecast = valid_data['naive']


r_squared = r2_score(actual, forecast)

r_squared



print(round(r_squared,6))

# Adjusted R²
n = len(actual)
p = 0  # number of predictors in naive forecast
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("R²:", round(r_squared, 10))
print("Adjusted R²:", round(adj_r_squared, 10))

##################################   trend only    ##############################

from statsmodels.formula.api import ols

#Make a column t which is just the number of the row 1,2,3,4...etc.

df_extended['t'] = range(1, len(df_extended) + 1)

df_extended.dtypes


#count has to be numeric (float (float) even if it has no decimals (is integer), in order to work for 

df_extended["count"] = df_extended['count'].astype(float)
 
df_extended

#ignores rows foro Nan values in count
model=ols("count ~ t", data=df_extended).fit()

model.summary()

predictions=model.predict()

#In order to fill column in df_extended with the predictions it must first match the length
# Create a full-length array with NaN
full_predictions = np.empty(len(df_extended))
full_predictions[:] = np.nan  # fill with NaN

# Fill the first part with your actual predictions
full_predictions[:len(predictions)] = predictions

full_predictions

# Assign to the new column

df_extended['trend only Predictions'] = full_predictions



plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals")
plt.plot(df_extended['Date of Admission'],df_extended['trend only Predictions'], label= "Trend Forecast")
plt.legend()

plt.show()

df_extended.dtypes

########################  Seasonality Only   #####################################



df_extended['month'] = df_extended['Date of Admission'].dt.month


month_dummies = pd.get_dummies(df_extended['month'], prefix='month')
month_dummies = month_dummies.drop(columns='month_12')  # drop December manually
month_dummies = month_dummies.astype(int)  # convert True/False → 1/0
df_extended= pd.concat([df_extended, month_dummies], axis=1)


df_extended.columns


seasonalmodel=ols('count ~ month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',
                  data=df_extended).fit()

seasonalmodel.summary()

predictions=seasonalmodel.predict()

#In order to fill column in df_extended with the predictions it must first match the length
# Create a full-length array with NaN
full_predictions = np.empty(len(df_extended))
full_predictions[:] = np.nan  # fill with NaN

# Fill the first part with your actual predictions
full_predictions[:len(predictions)] = predictions

full_predictions

# Assign to the new column
df_extended['seasonality only Predictions']=full_predictions

#plot withseasonality and actual count
plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals")
#plt.plot(df_extended['Date of Admission'],df_extended['trend only Predictions'], label= "Trend Forecast")
plt.plot(df_extended['Date of Admission'],df_extended['seasonality only Predictions'], label= "Seasonality Forecast")

plt.legend()
plt.show()

df_extended

#Forecast into the future month by a year (already made the rows for a year extra)


#Method one
# --- 5. Compute manual predictions for all available rows ---
df_extended['manual_pred'] = seasonalmodel.params['Intercept']


for col in seasonalmodel.params.index:
    if col.startswith('month_'):
        df_extended['manual_pred'] += df_extended[col].fillna(0) * seasonalmodel.params[col]


df_extended['manual_pred']

#method 2

df_extended['manual_pred_method2']= seasonalmodel.predict(df_extended)


df_extended[['manual_pred','manual_pred_method2']]

#plot with actual count, seasonality and forecast of future predictions (seasonality model)

plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals")
#plt.plot(df_extended['Date of Admission'],df_extended['trend only Predictions'], label= "Trend Forecast")
plt.plot(df_extended['Date of Admission'],df_extended['manual_pred'], label= "Seasonality Forecast with Predictions")
plt.legend()

plt.show()


#################### Seasonality and Trend  ##################################

seasonalandtrendmodel=ols('count ~ t+ month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',
                          data=df_extended).fit()

seasonalandtrendmodel.summary()

predictions=seasonalandtrendmodel.predict()



#In order to fill column in df_extended with the predictions it must first match the length
# Create a full-length array with NaN
full_predictions = np.empty(len(df_extended))
full_predictions[:] = np.nan  # fill with NaN

# Fill the first part with your actual predictions
full_predictions[:len(predictions)] = predictions

full_predictions

# Assign to the new column
df_extended['Seasonality & Trend Predictions'] = full_predictions

###
# Run prediction on the entire df_extended (including future months)

df_extended['Seasonality & Trend Forecast'] = seasonalandtrendmodel.predict(df_extended)



####


plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals",marker='o')
plt.plot(df_extended['Date of Admission'],df_extended['Seasonality & Trend Forecast'], label= "Seasonality & Trend Forecast",marker='x')
plt.legend()
plt.show()

#manual R squared and Adjusted to see if it is the same as model summary
valid_data = df_extended.dropna(subset=['count', 'Seasonality & Trend Predictions'])


pd.set_option('display.max_rows', None)

valid_data[['count','Seasonality & Trend Predictions']]


#reset back

pd.reset_option('display.max_rows')
    
# Step 2: Get actual values and forecasted values

actual = valid_data['count']
forecast = valid_data['Seasonality & Trend Predictions']


r_squared = r2_score(actual, forecast)

r_squared

# Adjusted R²
n = len(actual)
p = 12 # number of predictors in naive forecast
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("R²:", round(r_squared, 10))
print("Adjusted R²:", round(adj_r_squared, 10))

#Your model explains about 33% of the variance in count.

#Adjusted R² is much lower because you have 12 predictors and only 61 observations, so it penalizes the model for 
# its complexity.

#lets try to filter the first and last data values as they are outliers

from scipy.stats import iqr

counts=df_extended['count'].dropna()

counts

interquartilerange=iqr(counts)

interquartilerange

lower_thresh=np.quantile(counts,0.25)-1.5*interquartilerange

lower_thresh

upper_thresh=np.quantile(counts,0.75)+1.5*interquartilerange

upper_thresh

outliers=counts[(counts<lower_thresh)|(counts>upper_thresh)]

outliers
    
# Dropping the outliers count rows

df_no_outliers = df_extended[~df_extended['count'].isin(outliers)]

#checking the model performance of the data without outliers (Note that there is no index for 0, 33,
# or 60 since those are the outliers removed and we did not reset index)

df_no_outliers
    
#manual R squared and Adjusted to see if it is the same as model summary
valid_data = df_no_outliers.dropna(subset=['count', 'Seasonality & Trend Predictions'])


pd.set_option('display.max_rows', None)

valid_data[['count','Seasonality & Trend Predictions']]


#reset back

pd.reset_option('display.max_rows')
    
# Step 2: Get actual values and forecasted values

actual = valid_data['count']
forecast = valid_data['Seasonality & Trend Predictions']


r_squared = r2_score(actual, forecast)

r_squared

r_squared = 1 - (sum((actual - forecast)**2)/sum((actual - actual.mean())**2))

r_squared


# Adjusted R²
n = len(actual)
p = 12 # number of predictors in naive forecast
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("R²:", round(r_squared, 10))
print("Adjusted R²:", round(adj_r_squared, 10))

#The R square is negative beacuse just using the mean of the actual data would have been better I guess. 
#It got worse if you remove outliers because the outliers were initially included in model, you have to remove them 
# then run model, then asses performance.

seasonalandtrendmodel=ols('count ~ t+ month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',
                          data=df_no_outliers).fit()

seasonalandtrendmodel.summary()




#lets try a 3 month moving average, then a prediction model with seasonality, trend and 3mma as predictors.


########### just 3mma        ####################

df_extended['Three_monthMA'] = df_extended['count'].shift(1).rolling(window=3).mean()

df_extended[['count','Three_monthMA']]


plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals",marker='o')
plt.plot(df_extended['Date of Admission'],df_extended['Three_monthMA'], label= "3 month MA",marker='x')
plt.legend()
plt.show()


valid_data = df_extended.dropna(subset=['count', 'Three_monthMA'])


pd.set_option('display.max_rows', None)

valid_data[['count','Three_monthMA']]

    
# Step 2: Get actual values and forecasted values

actual = valid_data['count']
forecast = valid_data['Three_monthMA']


r_squared = r2_score(actual, forecast)

print('model with outliers')
r_squared

# Adjusted R²
n = len(actual)
p = 1 # number of predictors in naive forecast
adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

print("Model made with outliers")
print(adj_r_squared)


##check it with no outliers (do not just assess the model's performance without outliers if 
# you formed the model with outliers as that will always be worse, run/make the model again
# without outliers in data)

# Dropping the outliers count rows

df_no_outliers = df_extended[~df_extended['count'].isin(outliers)]

df_no_outliers['Three_monthMA'] = df_no_outliers['count'].shift(1).rolling(window=3).mean()

df_no_outliers[['count','Three_monthMA']]


#manual R squared and Adjusted to see if it is the same as model summary
valid_data = df_no_outliers.dropna(subset=['count', 'Three_monthMA'])


pd.set_option('display.max_rows', None)

valid_data[['count','Three_monthMA']]

    
# Step 2: Get actual values and forecasted values

actual = valid_data['count']
forecast = valid_data['Three_monthMA']


r_squared = r2_score(actual, forecast)

r_squared

#in this case it is also worse without outliers but with 3mma it just depends how close the 3 months values are to each other



##### Prediction model with seasonality, trend and 3mma as predictors.

df_extended.columns

df_extended['Three_monthMA'] = df_extended['count'].shift(1).rolling(window=3).mean()



seasonal_trend_3MMA_model=ols('count ~ t+ Three_monthMA +month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',
                          data=df_extended).fit()

seasonal_trend_3MMA_model.summary()

#this model was much better when I used 3monthma without shifting it down thats because 
# it was using the average calculating part of the actual data. It had 0.788 adjusted r squared. 
#now that I actually shifted it down to only use the past three months it is only 0.127 adjusted R squared
predictions=seasonal_trend_3MMA_model.predict()

# Assign to the new column

#In order to fill column in df_extended with the predictions it must first match the length
# Create a full-length array with NaN
full_predictions = np.empty(len(df_extended))
full_predictions[:] = np.nan  # fill with NaN

# Fill the first part with your actual predictions
full_predictions[:len(predictions)] = predictions

full_predictions

# Assign to the new column

df_extended['seasonal_trend_3MMA_model'] = full_predictions


df_extended[['count','seasonal_trend_3MMA_model']]

plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals",marker='o')
plt.plot(df_extended['Date of Admission'],df_extended['seasonal_trend_3MMA_model'], label= "seasonal_trend_3MMA_model",marker='x')
plt.legend()
plt.show()



#Note that you cannot forecast into future unless you add values into 3monthma for the future year but you are assuming
df_extended[['Three_monthMA','Seasonality & Trend Forecast']]

for i in range(len(df_extended)):
    if pd.isna(df_extended.loc[i, 'Three_monthMA']):
        # simple approach: use mean of last 3 seasonality & trend forecasted values
        last_vals = df_extended['count'].iloc[i-3:i].fillna(df_extended['Seasonality & Trend Forecast'].iloc[i-3:i])
        df_extended.loc[i, 'Three_monthMA'] = last_vals.mean()
        
df_extended['seasonal_trend_3MMA_model_with_forecast'] = seasonal_trend_3MMA_model.predict(df_extended)


plt.plot(df_extended['Date of Admission'], df_extended['count'],label="Monthyl Patient Totals",marker='o')
plt.plot(df_extended['Date of Admission'],df_extended['seasonal_trend_3MMA_model_with_forecast'], label= "seasonal_trend_3MMA_model_with_forecast",marker='x')
plt.legend()
plt.show()

####lets try the model without outliers

df_no_outliers = df_extended[~df_extended['count'].isin(outliers)]

df_no_outliers['Three_monthMA'] = df_no_outliers['count'].shift(1).rolling(window=3).mean()


seasonal_trend_3MMA_no_outlier_model=ols('count ~ t+ Three_monthMA +month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',
                          data=df_no_outliers).fit()

seasonal_trend_3MMA_no_outlier_model.summary()
 
#it is better with an R square of 0.468 as opposed to 0.326 but it says there is multicolinearity


########### Try with a lag term or three lag terms as predictors with trend and dummy months instead of 3mma

df_extended['lag1'] = df_extended['count'].shift(1)
df_extended['lag2'] = df_extended['count'].shift(2)
df_extended['lag3'] = df_extended['count'].shift(3)



lag_trend_dummy_model=ols('count ~ t+ lag1 +lag2 +lag3 +month_1+ month_2+ month_3+ month_4 +month_5+ month_6+month_7+ month_8+ month_9+ month_10+ month_11',
                          data=df_extended).fit()

lag_trend_dummy_model.summary()


###try exponential smoothing



from statsmodels.tsa.holtwinters import ExponentialSmoothing

hw_model = ExponentialSmoothing(df_extended['count'], trend='add', seasonal='add', seasonal_periods=12).fit()
df_extended['predicted_hw'] = hw_model.fittedvalues


#manual R squared and Adjusted to see if it is the same as model summary
valid_data = df_extended.dropna(subset=['count', 'predicted_hw'])


pd.set_option('display.max_rows', None)

valid_data[['count','predicted_hw']]

    
# Step 2: Get actual values and forecasted values

actual = valid_data['count']
forecast = valid_data['predicted_hw']


r_squared = r2_score(actual, forecast)

r_squared



######trying with prophet
#pip install prophet

from prophet import Prophet

df_extended.dtypes

#prophet requires date to be named ds and the values you want to predict as y
prophet_df = df_extended[['Date of Admission', 'count']].rename(columns={'Date of Admission': 'ds', 'count': 'y'})
prophet_df = prophet_df.dropna()  # drop NaNs

model = Prophet(
    yearly_seasonality=True,  # because your data has period 12 (monthly)
    weekly_seasonality=False,
    daily_seasonality=False
)
model.fit(prophet_df)

forecast = model.predict(prophet_df)


# Assign to the new column

#In order to fill column in df_extended with the predictions it must first match the length
# Create a full-length array with NaN
full_predictions = np.empty(len(df_extended))
full_predictions[:] = np.nan  # fill with NaN

# Fill the first part with your actual predictions
full_predictions[:len(forecast['yhat'])] = forecast['yhat']

full_predictions

# Assign to the new column

df_extended['predicted_prophet'] = full_predictions

df_extended[['count','predicted_prophet']]


y_true = prophet_df['y']
y_pred = forecast['yhat']

r2 = r2_score(y_true, y_pred)
print("R-squared:", round(r2, 3))




##################  Resample to monthly              ##############################
########### length of stay averages per month         ##########


# 3 ways to resample 

stay_trend = healthcaredf.resample('M', on='Date of Admission')['Length of Stay (days)'].mean()

stay_trend

stay_trend = healthcaredf.set_index('Date of Admission')['Length of Stay (days)'].resample('M').mean()

stay_trend

stay_trend = (
    healthcaredf.groupby(pd.Grouper(key='Date of Admission', freq='M'))['Length of Stay (days)']
      .mean().reset_index(name='Monthly Average Stay'))

stay_trend

########## Count of patients per month


df_time = healthcaredf.groupby('Date of Admission').size().resample('M').sum()
df_time



df_time.plot(title='Monthly Admissions Trend')


########  Billing sum per month

billing_sum_per_month = healthcaredf.resample('M', on='Date of Admission')['Billing Amount'].sum()
billing_sum_per_month

#method 2 

monthly_billing_sum = (
    healthcaredf.groupby(pd.Grouper(key='Date of Admission', freq='M'))['Billing Amount']
      .sum().reset_index(name='Monthly Billing Sum'))

billing_sum_per_month.plot(title='Monthly Billing Sum')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.show()

#billing averages per month
billing_month_average = healthcaredf.resample('M', on='Date of Admission')['Billing Amount'].mean()
billing_month_average


billing_month_average.plot(title='Monthly Billing Average per Month')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.show()



# ########### count of patients monthly per admission type        ############
df = healthcaredf.copy()

df.info()

df.columns

monthly_admissions= df.set_index('Date of Admission').resample('M')['Admission Type'].value_counts().unstack(fill_value=0)

monthly_admissions.reset_index()


#or without setting index manually (still sets it I think)

monthly_admissions = df.resample('M', on='Date of Admission')['Admission Type'].value_counts().unstack(fill_value=0)


monthly_admissions

type(monthly_admissions)

monthly_admissions.columns

monthly_admissions['Elective']


plt.plot(monthly_admissions.index,  monthly_admissions['Elective'], label='Elective')
plt.plot(monthly_admissions.index,monthly_admissions['Emergency'], label='Emergency')
plt.plot(monthly_admissions.index,monthly_admissions['Urgent'], label='Urgent')
plt.title("Count of Patients Monthly per admission type")
plt.xlabel("Date")
plt.ylabel("Number of Patients")
plt.legend()
plt.show()

############## Count of Patients Monthly Per Age Group  ############################

df.columns
monthly_count_per_age_group= df.set_index('Date of Admission').resample('M')['Age Group'].value_counts().unstack(fill_value=0)

monthly_count_per_age_group



plt.plot(monthly_count_per_age_group.index,  monthly_count_per_age_group['Teens(10-17)'], label='Teens (10-17)')
plt.plot(monthly_count_per_age_group.index,  monthly_count_per_age_group['Young Adults(18-24)'], label='Young Adults(18-24)')
plt.plot(monthly_count_per_age_group.index,  monthly_count_per_age_group['Adults(25-64)'], label='Adults(25-64)')
plt.plot(monthly_count_per_age_group.index,  monthly_count_per_age_group['Seniors(>65)'], label='Seniors(>65)')
plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1,1))
plt.xlabel("Date")
plt.ylabel("Number of Patients")
plt.title("Count of Patients Monthly per Age Group")

plt.show()



###########   Medical Condition Patients Trend Analysis      #############################

monthly_diseases=df.resample('M', on='Date of Admission')['Medical Condition'].value_counts().unstack(fill_value=0)

monthly_diseases




### option 1 (subplots)


n_cols = 3
n_rows = (len(monthly_diseases.columns) + n_cols - 1) // n_cols

n_rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows*3), sharex=True, sharey=True)

for ax, condition in zip(axes.flatten(), monthly_diseases.columns):
    ax.plot(monthly_diseases.index, monthly_diseases[condition])
    ax.set_title(condition)
    
plt.tight_layout()
plt.show()

#option 2 subplots also
monthly_diseases.columns

monthly_diseases.plot(
    subplots=True,
    layout=(3, 2),  # 3 rows × 2 columns for the 6 medical conditions
    figsize=(12, 6),
    sharex=True,
    sharey=True)
plt.tight_layout()
plt.show()


#option 3 but force the legend to be on lower left

axes = monthly_diseases.plot(
    subplots=True,
    layout=(3, 2),
    figsize=(12, 6),
    sharex=True,
    sharey=True
)

# Force all legends to bottom left
for ax in axes.flatten():
    ax.legend(loc='lower left')

plt.suptitle('Monthly Disease Trends', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

#option 3 again but with titles instead of legend and 2 columns 3 rows

axes = monthly_diseases.plot(
    subplots=True,
    layout=(3, 2),
    figsize=(12, 6),
    sharex=True,
    sharey=True,
    legend=False
)

for ax, col in zip(axes.flatten(), monthly_diseases.columns):
    ax.set_title(col, fontsize=10)

plt.suptitle('Monthly Disease Trends', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()




############ Resample but aggregate with one column, grouped by another column #####

#### Monthly Billing Sum Per Medical Condition   ############################

#method one (with apply and lambda)

monthly_billing_sum_per_disease = (df.resample('M', on='Date of Admission')
    .apply(lambda x: x.groupby('Medical Condition')['Billing Amount'].sum()))

monthly_billing_sum_per_disease


#method 2 without apply and lambda

monthly_billing_sum_per_disease = (df.groupby([
        pd.Grouper(key='Date of Admission', freq='M'),'Medical Condition'])['Billing Amount']
        .sum().unstack(fill_value=0))

monthly_billing_sum_per_disease

#How high is monthly billing sum per medical condition correlated with monthly count of patients per medical condition?

monthly_diseases['Arthritis'].corr(monthly_billing_sum_per_disease['Arthritis'])

monthly_billing_sum_per_disease['Arthritis'].corr(monthly_diseases['Arthritis'])


healthcaredf['Hospital'].value_counts()



############### practice summarizing specific hospitals or diseases or doctors ############


#### doctor

healthcaredf['Doctor'].value_counts().head()

#Filtering data for doctor michael smith
Michael_Smith=healthcaredf[healthcaredf['Doctor']=='Michael Smith']

Michael_Smith

#Billing Sum for michael smith
Michael_Smith['Billing Amount'].sum()

#method 2 for billing sum of doctor michael_smith

healthcaredf[healthcaredf['Doctor']=='Michael Smith']['Billing Amount'].sum()

#number of patients for doctor micheal smith (27)
len(Michael_Smith)

#number of medications prescribed by doctor michael smith (aspirin 7 times, ibuprofen 7 times,
# paracetamol 7 times, lipitor 3 times, penicillin 3 times )

healthcaredf[healthcaredf['Doctor']=='Michael Smith']['Medication'].value_counts()

#method 2 for num of medications prescribed by doctor michael smith

Michael_Smith['Medication'].value_counts()


#Just realized there are no NA rows so every patient was prescribed a medication

healthcaredf.info()

####  specific hospital sum

healthcaredf['Hospital'].value_counts().head()

LLC_Smith_hospital=healthcaredf[healthcaredf['Hospital']=='LLC Smith']

LLC_Smith_hospital

LLC_Smith_hospital['Billing Amount'].sum()

## formatted with commas and 2 decimals

#method 1
f"{LLC_Smith_hospital['Billing Amount'].sum():,.2f}"

#method 2
total = LLC_Smith_hospital['Billing Amount'].sum()
print(format(total, ",.2f"))

#formatted with $ sign and commas and 2 decimals

f"${LLC_Smith_hospital['Billing Amount'].sum():,.2f}"

total = LLC_Smith_hospital['Billing Amount'].sum()

print("${:,.2f}".format(total))

healthcaredf['Hospital'].value_counts()[healthcaredf['Hospital'].value_counts() > 10]

healthcaredf['Hospital'].value_counts()[healthcaredf['Hospital'].value_counts() > 10].index

healthcaredf[healthcaredf['Hospital'].value_counts() > 10]

healthcaredf.columns 

######## correlation heat map to determine any models that should be formed

healthcaredf.info()
#numerical types are age, billing amount, room number,length of stay (Days)

plt.figure(figsize=(10, 8))
correlation_matrix = healthcaredf[['Age', 'Billing Amount', 'Room Number','Length of Stay (days)']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()

#nothing with high correlation 


### Trying to make a model that predicts billing amount based of of predicotrs (medical condition, age,     #####
#gender, blood type, date of admission, length of stay, test results insurance type)                        ######


healthcaredf.info()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from sklearn.compose import ColumnTransformer

##Step 1 select features and target  ##


#first take out the rows where billing amount is 0 or less, in case I want to use the log later 

df = healthcaredf.copy()

# Remove rows where Billing Amount is <= 0 (make sure you dont run model without doing this)


df = df[df['Billing Amount'] > 0]


# Target
y = df['Billing Amount']

# Predictors (features) (do not use hospital since that has too many 39,815 its more of a primary key)
x_variables= df[['Age','Gender','Blood Type','Medical Condition',
                  'Insurance Provider','Admission Type','Length of Stay (days)',
                  'Age Group']]

## Step 2 split/train test  (20 percent of the data is test, 80 percent is training ##

#splits to x_ and y_ train and test variables where y is billing amount and x is predictors
# (columns from  x_variables and their data) 

x_train, x_test, y_train, y_test = train_test_split(
   x_variables, y, test_size=0.2, random_state=42)


x_train.head()
y_train.head()

y.head(10)

df['Billing Amount'].head()


##Step 3 turn categorical data into numbers ("hot encode categorical variables") ##


categorical_cols = x_train.select_dtypes(['category','object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough')



# Fit encoder on training data, test data, and one for all of it 
#note that the  fit_transform learns the cateogiries + creates dummy mapping and transforms to x_train
#the test data uses the mapping learned from training and ignores the categories that were not one the training set
#thats why it is just transform and not fit_transform on x_test
#note that the test performance is the only valid performance metric since that is simulating the future unkown data

x_train_encoded = preprocessor.fit_transform(x_train)

x_test_encoded = preprocessor.transform(x_test)



## Step 4 is optional log transform target ##

#use np.exm1(pred) to convert predictions back to dollars later
y_train_log = np.log1p(y_train)  # log(1 + y)
y_test_log = np.log1p(y_test)

## Step 5 train gradient boosting regressor ##

# Choose whether to use log or raw target. Billing Amounts often have a long right tail (a few very high bills,
# most moderate) and so using log helps.

use_log = True  # set to False if you want raw, True if you want log

if use_log:
    y_train_target = y_train_log
else:
    y_train_target = y_train


#build 500 small trees, each tree can only split up to 4
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,
                                  max_depth=4, random_state=42)


#remember if you did not log, y_train_target is the same as y_train

#now the learning step: take encoded predictors and take target ys, learns patterns by building 500 weak trees
#that collectively form a strong predictor

model.fit(x_train_encoded, y_train_target)


#note that rpart regression tree is 1 tree, random forest is hundreds, gradient boosting is hundreds sequential trees

#view the first tree (the one that explains the most variance)
#note that it does not make sense since it is encoded 

estimator = model.estimators_[0][0]
from sklearn.tree import export_text
print(export_text(estimator))



## Step 6 evaluate model ## 

# Predictions
y_pred_train = model.predict(x_train_encoded)

y_pred_test = model.predict(x_test_encoded)

y_pred_test

y_pred_train

#convert log billing back to normal if you converted and even if you did not
#it renames to y_test_actual


if use_log:
    y_pred_test = np.expm1(y_pred_test)
    y_pred_train = np.expm1(y_pred_train)


#By using y_test_actual everywhere in your metric calculations, you can write the same code for RMSE, R², etc.
# without having to branch for whether you used log transform or not.


#root mean squared error gives the square root of mse, so it gives the average magnitute error in the 
#same unit as your target (y dollars), lower rmse = better fit.
#if you put squared =true its just mse. false= RMSE 

rmse = mean_squared_error(y_test, y_pred_test, squared=False)
print(f'RMSE on test set: ${rmse:,.2f}')

#get r square

#method 1 for 
model.score(x_test_encoded, y_test)

model.score(x_train_encoded, y_train)


#method 2

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


print("TRAIN R2:", r2_score(y_train, y_pred_train))

print("TEST R2:", r2_score(y_test_actual, y_pred_test))


#performed terribly. (Without using log y: R2 of 0.035 for train and -0.007 for test. 
# with log for y its worse at -0.12106 for train and -0.163 for test
### view differences and manually calculate r2

compare_train=pd.DataFrame({'actual':y_train,'predictions':y_pred_train})

compare_train.head()

y_actual = y_train
y_pred   = y_pred_train

#mean of actuals

y_mean = y_actual.mean()

#residual sum of squaress

ss_res = ((y_actual - y_pred) ** 2).sum()

#total sum of squares
ss_tot = ((y_actual - y_mean) ** 2).sum()

#r square
r2_manual = 1 - (ss_res / ss_tot)
print("Manual R²:", r2_manual)






## Step 7 get feature importance ##

# Get feature names after one-hot encoding (make sure you last run the tree model and not another type,
# since the "model" is updated, but also fit the model)

onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)

numeric_cols = [col for col in x_variables.columns if col not in categorical_cols]

feature_names = np.concatenate([onehot_feature_names, numeric_cols])

importances = model.feature_importances_

indices = np.argsort(importances)[::-1]

# Plot top 20
plt.figure(figsize=(12,6))
plt.bar(range(20), importances[indices][:20])
plt.xticks(range(20), feature_names[indices][:20], rotation=90)
plt.title('Top 20 Feature Importances')
plt.show()

#############
####gonna try this again but with a multiple regresion

import statsmodels.api as sm
  # after simplification + one hot encoding

# select numeric columns
numeric_cols = ['Age','Length of Stay (days)']  # plus any numeric features

# select categorical columns
cat_cols = ['Gender','Blood Type','Medical Condition','Insurance Provider','Admission Type','Age Group']

# convert categorical to dummies (with drop first=true, the first dummy is the all zero for categorical)
x_encoded = pd.get_dummies(x_variables[numeric_cols + cat_cols], drop_first=True) 

x_variables.head()

x_encoded = sm.add_constant(x_encoded)  # intercept

#maybe convert to ones and zeroes from true and false

#
x_encoded = x_encoded.astype(int)

x_encoded.head()

df['Blood Type'].value_counts()



model = sm.OLS(y, x_encoded).fit()

print(model.summary())
w
# Get predictions
y_pred = model.predict(x_encoded) 

y_pred

#performs terrbily, maybe do variable selection procedures.