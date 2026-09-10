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
#check outliers in the data using iqr method
num_features = df.select_dtypes(include='number')
Q1 = num_features.quantile(0.25)
Q3 = num_features.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (num_features < lower_bound) | (num_features > upper_bound)
print("Number of outliers in each column:\n", outliers.sum())
#only 24 outliers in the data out of 5000 rows, we can ignore them for now, but we can also remove them if needed
#data cleaning remove duplicate values, missing values
#Two real issues to fix here — everything else in this dataset is already clean, so we don't manufacture cleaning steps that aren't needed.
#the rest of that student's data (age, study hours, stress level, etc.) is still valid and useful — throwing away the whole row over one bad value wastes good data. Clipping caps the impossible value at the nearest realistic one (0 hours) without discarding everything else about that student.
print(df.describe())
#Converting negative or unrealitstic value to realistic value
df['Physical_Activity_Hours'] = df['Physical_Activity_Hours'].clip(lower=0)
#We're not dropping any columns either — every column here has a plausible reason to matter for predicting mental health, and we already confirmed none of them are empty or constant.
print(df.describe())
print(df.shape)
#Skewness
#What skewness is: a measure of how lopsided a numeric column's distribution is. A value near 0 means roughly symmetric (bell-shaped); a large positive or negative value means the data leans heavily to one side, with a long tail.
#Why it matters: models like Linear Regression assume features are roughly well-behaved. A heavily skewed column (think: a long tail of extreme values) can quietly drag the model's predictions in that direction. Tree-based models like Random Forest don't care about skew — but since we're also training a Linear Regression baseline, it's worth fixing.
num_cols = df.select_dtypes(include='number')
num_cols.skew()
# near to 0 -> Centralized (0.01, 0.002)
# negative -> Left Skewed (-1.56)
# poistive (greater than 0) -> Right Skewed (1.256)
#feature engineering
#Feature Engineering 
#Country has 111 unique values in this dataset — one-hot encoding that directly would add 110+ mostly-empty columns, which hurts the model far more than it helps (this is called high cardinality).
#Dropping Country entirely throws away real signal — a student's country genuinely correlates with things like internet access, culture, and sleep norms
#The fix: keep the top 10 most frequent countries as their own category, and bucket everything else into "Other". We keep the signal that matters and lose the noise that doesn't.
top_countries = df['Country'].value_counts().index[:10].tolist()
def group_countries(country):
  if country in top_countries:
    return country
  else:
    return 'Other'
df['Grouped_country'] = df['Country'].apply(group_countries)
print(df['Grouped_country'].value_counts())
