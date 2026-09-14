import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.tools import numeric
#load data
df= pd.read_csv('F:\\project\\menal health prediction\\Student Social Media And Mental Health Impact.csv')
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
#Train test split
from sklearn.model_selection import train_test_split
#encoding strategy
#Before we jump into code, let's decide how each categorical column should be encoded — this decision matters more than the code itself.
#Stress_Level → Ordinal Encoding. Its categories have a real, meaningful order: Low < Medium < High < Very High. We already saw in EDA (section 4.3) that the score drops step by step as stress increases — encoding it as 0, 1, 2, 3 preserves that order for the model.
#Gender, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Country_Grouped → One-Hot Encoding. These categories have no natural order — "Instagram" isn't "greater than" "LinkedIn". One-hot encoding creates a separate 0/1 column per category so the model doesn't accidentally assume a false ranking.
print(df.columns)

schewed_col=["Study_Hours"]
numeric_col=["Age","Avg_Daily_Usage_Hours","Daily_Unlocks",'Physical_Activity_Hours', 'Sleep_Hours_Per_Night']
oridinal_col=["Stress_Level",]
normal_col=["Gender","Academic_Level","Most_Used_Platform", "Purpose_Of_Use", "Grouped_country"]

feature_col=numeric_col+oridinal_col+normal_col+schewed_col
x=df[feature_col]
y=df["Mental_Health_Score"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)



#Preprocessing using ColumnTransformer
#Our columns need different treatment:
#Study_Hours (the skewed one) → impute → log1p transform → scale
#The other numeric columns → impute → scale (no skew to fix)
#Stress_Level → impute → OrdinalEncoder with an explicit order
#Categorical columns with no natural order → impute → OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV


#skew pipeline
skew_pipeline=Pipeline(steps=[
    ("log1p",FunctionTransformer(np.log1p)),
    ("scale",StandardScaler())
])
#other number columns pipeline
number_pipeline=Pipeline(steps=[
    ("scale",StandardScaler())
])
#ordinal pipeline
oridinal_pipeline=Pipeline(steps=[
    ("ordinal",OrdinalEncoder(categories=[["Low","Medium","High","Very High"]]))
])
#nominal pipeline
nominal_pipeline=Pipeline(steps=[
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])

preprocessor=ColumnTransformer(transformers=[
    ("skew",skew_pipeline,schewed_col),
    ("number",number_pipeline,numeric_col),
    ("oridinal",oridinal_pipeline,oridinal_col),
    ("nominal",nominal_pipeline,normal_col)
])


#Baseline: Linear Regression
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

lr_pipeline.fit(x_train,y_train)
lr_pred = lr_pipeline.predict(x_test)
lr_pred_train = lr_pipeline.predict(x_train)

lr_r2_train=r2_score(y_train,lr_pred_train)
lr_r2_test=r2_score(y_test,lr_pred)
lr_mae=mean_absolute_error(y_test,lr_pred)
lr_mse=mean_squared_error(y_test,lr_pred)
lr_rmse=np.sqrt(lr_mse)

print("R2 score on training:",lr_r2_train)
print("R2 score on testing:",lr_r2_test)
print("MAE:",lr_mae)
print("MSE:",lr_mse)
print("RMSE:",lr_rmse)

#Random Forest (default settings)

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

rf_pipeline.fit(x_train,y_train)
rf_pred = rf_pipeline.predict(x_test)
rf_pred_train = rf_pipeline.predict(x_train)

rf_r2_train=r2_score(y_train,rf_pred_train)
rf_r2_test=r2_score(y_test,rf_pred)
rf_mae=mean_absolute_error(y_test,rf_pred)
rf_mse=mean_squared_error(y_test,rf_pred)
rf_rmse=np.sqrt(rf_mse)

print("R2 score on training:",rf_r2_train)
print("R2 score on testing:",rf_r2_test)
print("MAE:",rf_mae)
print("MSE:",rf_mse)
print("RMSE:",rf_rmse)

#Hyper parameter tunning on Random forest

param_grid={
    "regressor__n_estimators":[500,700,900],
    "regressor__max_depth":[60,70,80],
    "regressor__min_samples_split":[2,5,10],
    "regressor__min_samples_leaf":[1,2,3]
}

Random_search=RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

Random_search.fit(x_train,y_train)

print(Random_search.best_params_)
print(Random_search.best_score_)

#XGboost train default
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

# Train
xgb_pipeline.fit(x_train, y_train)

# Prediction
xgb_pred = xgb_pipeline.predict(x_test)
xgb_pred_train = xgb_pipeline.predict(x_train)

# Training R2
xgb_r2_train = r2_score(y_train, xgb_pred_train)

# Testing R2
xgb_r2_test = r2_score(y_test, xgb_pred)

# Other metrics
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_rmse = np.sqrt(xgb_mse)

print("XGBoost Default Model")
print("----------------------")
print("R2 score on training:", xgb_r2_train)
print("R2 score on testing:", xgb_r2_test)
print("MAE:", xgb_mae)
print("MSE:", xgb_mse)
print("RMSE:", xgb_rmse)


#tunnig xg model with random search cv

xgb_param_grid = {
    
    "regressor__n_estimators": [100, 200, 300, 500, 700],
    
    "regressor__max_depth": [3, 4, 5, 6, 8, 10],
    
    "regressor__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    
    "regressor__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    
    "regressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    
    "regressor__min_child_weight": [1, 3, 5, 7],
    
    "regressor__gamma": [0, 0.1, 0.2, 0.5, 1]
}

xgb_random_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_param_grid,
    n_iter=30,
    cv=5,
    scoring="r2",
    verbose=2,
    n_jobs=-1,
    random_state=42
)

xgb_random_search.fit(x_train, y_train)

print(xgb_random_search.best_params_)
print(xgb_random_search.best_score_)

best_xgb = xgb_random_search.best_estimator_

# Predictions
xgb_tuned_pred = best_xgb.predict(x_test)
xgb_tuned_pred_train = best_xgb.predict(x_train)

# Metrics
xgb_tuned_r2_train = r2_score(y_train, xgb_tuned_pred_train)
xgb_tuned_r2_test = r2_score(y_test, xgb_tuned_pred)

xgb_tuned_mae = mean_absolute_error(y_test, xgb_tuned_pred)
xgb_tuned_mse = mean_squared_error(y_test, xgb_tuned_pred)
xgb_tuned_rmse = np.sqrt(xgb_tuned_mse)

print("XGBoost Tuned Model")
print("-------------------")
print("R2 on training:", xgb_tuned_r2_train)
print("R2 on testing :", xgb_tuned_r2_test)
print("MAE           :", xgb_tuned_mae)
print("MSE           :", xgb_tuned_mse)
print("RMSE          :", xgb_tuned_rmse)

print("R2 gap:", xgb_tuned_r2_train - xgb_tuned_r2_test)


#train model with xg boost with tunning optuna

import optuna
from sklearn.model_selection import cross_val_score

# 1. Objective Function
def objective(trial):

    xgb_model = XGBRegressor(

        # Number of boosting rounds
        n_estimators=trial.suggest_int(
            "n_estimators",
            500,
            1000,
            step=50
        ),

        # Learning rate
        learning_rate=trial.suggest_float(
            "learning_rate",
            0.015,
            0.06,
            log=True
        ),

        # Tree depth
        max_depth=trial.suggest_int(
            "max_depth",
            5,
            10
        ),

        # Minimum child weight
        min_child_weight=trial.suggest_int(
            "min_child_weight",
            1,
            5
        ),

        # Row sampling
        subsample=trial.suggest_float(
            "subsample",
            0.80,
            1.0
        ),

        # Feature sampling
        colsample_bytree=trial.suggest_float(
            "colsample_bytree",
            0.80,
            1.0
        ),

        # Minimum loss reduction
        gamma=trial.suggest_float(
            "gamma",
            0.0,
            0.3
        ),

        # L1 regularization
        reg_alpha=trial.suggest_float(
            "reg_alpha",
            0.0,
            0.5
        ),

        # L2 regularization
        reg_lambda=trial.suggest_float(
            "reg_lambda",
            0.05,
            5.0,
            log=True
        ),

        random_state=42,
        n_jobs=-1
    )
    # Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", xgb_model)
    ])
    # 5-Fold Cross Validation
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )
    return scores.mean()

# 2. Create Optuna Study
study_focused = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(
        seed=42
    )
)

# 3. Run Optuna
study_focused.optimize(
    objective,
    n_trials=50,
    show_progress_bar=True
)

# 4. Best Results
print("\n" + "=" * 60)
print("OPTUNA FOCUSED SEARCH RESULTS")
print("=" * 60)

print("\nBest CV R2:")
print(f"{study_focused.best_value:.6f}")

print("\nBest Parameters:")

for parameter, value in study_focused.best_params.items():
    print(f"{parameter}: {value}")

# 5. Create Final XGBoost
best_xgb_optuna = XGBRegressor(
    **study_focused.best_params,
    random_state=42,
    n_jobs=-1
)

# 6. Final Pipeline
optuna_focused_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", best_xgb_optuna)
])

# 7. Train
optuna_focused_pipeline.fit(
    x_train,
    y_train
)

# 8. Predictions
train_pred = optuna_focused_pipeline.predict(x_train)

test_pred = optuna_focused_pipeline.predict(x_test)



# 9. Evaluation
train_r2 = r2_score(
    y_train,
    train_pred
)

test_r2 = r2_score(
    y_test,
    test_pred
)

mae = mean_absolute_error(
    y_test,
    test_pred
)

mse = mean_squared_error(
    y_test,
    test_pred
)

rmse = np.sqrt(mse)
r2_gap = train_r2 - test_r2

# 10. Final Results
print("\n" + "=" * 60)
print("FINAL OPTUNA XGBOOST MODEL")
print("=" * 60)

print(f"\nTrain R2 : {train_r2:.6f}")
print(f"Test R2  : {test_r2:.6f}")
print(f"MAE      : {mae:.6f}")
print(f"MSE      : {mse:.6f}")
print(f"RMSE     : {rmse:.6f}")
print(f"R2 Gap   : {r2_gap:.6f}")

# 11. Compare Against Random Search

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

print(f"\nRandom Search Test R2 : 0.8973")
print(f"Optuna Test R2        : {test_r2:.4f}")

if test_r2 > 0.8973:
    print("\nOptuna performed BETTER than Random Search.")
else:
    print("\nRandom Search is still BETTER than Optuna.")


#evaluation 
# 1. Function to calculate regression metrics
def evaluate_model(model_name, model, X_train, y_train, X_test, y_test):

    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    # Training metrics
    train_r2 = r2_score(y_train, train_pred)
    # Testing metrics
    test_r2 = r2_score(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    mse = mean_squared_error(y_test, test_pred)
    rmse = np.sqrt(mse)
    # Overfitting gap
    r2_gap = train_r2 - test_r2
    return {
        "Model": model_name,
        "Train R²": train_r2,
        "Test R²": test_r2,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R² Gap": r2_gap
    }
# 2. Evaluate all models
results = []
# Linear Regression
results.append(
    evaluate_model(
        "Linear Regression",
        lr_pipeline,
        x_train,
        y_train,
        x_test,
        y_test
    )
)
# Random Forest - Default
results.append(
    evaluate_model(
        "Random Forest - Default",
        rf_pipeline,
        x_train,
        y_train,
        x_test,
        y_test
    )
)
# Random Forest - RandomizedSearchCV
best_rf = Random_search.best_estimator_
results.append(
    evaluate_model(
        "Random Forest - Tuned",
        best_rf,
        x_train,
        y_train,
        x_test,
        y_test
    )
)
# XGBoost - Default
results.append(
    evaluate_model(
        "XGBoost - Default",
        xgb_pipeline,
        x_train,
        y_train,
        x_test,
        y_test
    )
)
# XGBoost - RandomizedSearchCV
best_xgb = xgb_random_search.best_estimator_
results.append(
    evaluate_model(
        "XGBoost - Random Search",
        best_xgb,
        x_train,
        y_train,
        x_test,
        y_test
    )
)
# XGBoost - Optuna
results.append(
    evaluate_model(
        "XGBoost - Optuna",
        optuna_focused_pipeline,
        x_train,
        y_train,
        x_test,
        y_test
    )
)
# 3. Create comparison DataFrame
comparison_df = pd.DataFrame(results)
# 4. Round values
comparison_df = comparison_df.round(4)
# 5. Sort by Test R²
comparison_df = comparison_df.sort_values(
    by="Test R²",
    ascending=False
).reset_index(drop=True)
# 6. Display final table
print("=" * 90)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 90)

display(comparison_df)