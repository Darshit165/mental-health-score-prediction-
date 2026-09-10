import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.tools import numeric
#load data
df= pd.read_csv('D:\project\mental health score prediction\Student Social Media And Mental Health Impact.csv')
print(df.head())
#check data types and missing values, duplicate value , check statistics of the data
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
#drop duplicate values
df.drop_duplicates(inplace=True)
#EDA
#check the distribution of target mental health score
plt.figure(figsize=(7, 6))
sns.histplot(df['Mental_Health_Score'],kde=True)
plt.title('Distribution of Mental Health Score')
plt.xlabel('Mental Health Score')
plt.ylabel('Frequency')
plt.show()
#check the correlation between features and target variable
plt.figure(figsize=(7, 6))
sns.heatmap(df.corr(numeric_only="true"),annot=True)
plt.title('Correlation Heatmap')
plt.show()
#stress level vs mental health score
order = ['Low', 'Medium', 'High', 'Very High']
plt.figure(figsize=(7, 6))
sns.boxplot(x='Stress_Level',y='Mental_Health_Score',data=df,order=order)
plt.title('Stress Level vs Mental Health Score')
plt.xlabel('Stress Level')
plt.ylabel('Mental Health Score')
plt.show()
#Avg_Daily_Usage_Hours vs mental health score
plt.figure(figsize=(7, 6))
sns.scatterplot(x='Avg_Daily_Usage_Hours',y='Mental_Health_Score',data=df)
plt.title('Avg Daily Usage Hours vs Mental Health Score')
plt.xlabel('Avg Daily Usage Hours')
plt.ylabel('Mental Health Score')
plt.show()
#sleep hours vs mental health score
plt.figure(figsize=(7, 6))
sns.scatterplot(x='Sleep_Hours_Per_Night',y='Mental_Health_Score',data=df)
plt.title('Sleep Hours vs Mental Health Score')
plt.xlabel('Sleep Hours Per Night')
plt.ylabel('Mental Health Score')
plt.show()
