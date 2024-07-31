#!/usr/bin/env python
# coding: utf-8

# ## IMPORTING DATA AND PREPROCESSING

# In[1]:


import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly.subplots import make_subplots
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp, tpe, Trials, fmin
from scipy.stats import randint
import joblib


# In[2]:


# Establish a connection to your MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Oluwaseyi2000",
    database="hyandai"
)


# In[3]:


# Create a cursor object
cursor = conn.cursor()


# In[4]:


# Execute your SQL query
query = """
SELECT Date, Open, High, Low, Close, `Adj Close`, Volume
FROM hyandai_full
ORDER BY Date
"""
cursor.execute(query)


# In[5]:


# Fetch all rows from the result
data = cursor.fetchall()


# In[6]:


# Get column names
columns = [i[0] for i in cursor.description]


# In[7]:


# Close the cursor and connection
cursor.close()
conn.close()


# In[8]:


# Create a pandas DataFrame
df = pd.DataFrame(data, columns=columns)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


# In[9]:


df


# In[10]:


# Display descriptive statistics
print(df.describe())


# In[11]:


df.info()


# In[12]:


df.isnull().any()


# In[13]:


df.columns


# In[14]:


df2 = df.copy()
df3 = df.copy()


# In[ ]:





# In[ ]:





# ## EDA

# In[15]:


df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[ ]:





# In[16]:


# Visualize the distribution of the data
plt.figure(figsize=(10, 6))
sns.histplot(df['Close'], bins=30)
plt.title('Distribution of Closing Prices')
plt.show()


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

# Plot the closing price
plt.figure(figsize=(14, 7))
plt.plot(df['Close'], label='Closing Price')
plt.title('Hyundai Motor Company Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[18]:



# Time series decomposition
result = seasonal_decompose(df['Close'], model='additive', period=20)

# Adjust the figure size and layout
plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.5)


# Adjust the figure size and layout
fig, axes = plt.subplots(4, 1, figsize=(16, 12))
plt.subplots_adjust(hspace=0.5)

# Plot the decomposed components
axes[0].plot(result.observed, label='Observed')
axes[0].set_title('Time Series Decomposition')
axes[0].legend(loc='upper left')

axes[1].plot(result.trend, color='green', label='Trend')
axes[1].legend(loc='upper left')

axes[2].plot(result.seasonal, color='orange', label='Seasonal')
axes[2].legend(loc='upper left')

axes[3].plot(result.resid, color='red', label='Residual')
axes[3].legend(loc='upper left')

# Adjust the spacing and labels
plt.tight_layout()
plt.show()


# In[19]:


# Plot monthly seasonality
df['Month'] = df.index.month
monthly_avg = df.groupby('Month')['Close'].mean()

plt.figure(figsize=(10, 6))
monthly_avg.plot(kind='bar')
plt.title('Monthly Average Closing Prices')
plt.xlabel('Month')
plt.ylabel('Average Closing Price')
plt.show()


# In[20]:


# Calculate daily returns and volatility
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].rolling(window=252).std() * (252**0.5)  # Annualized volatility

# Plot volatility
plt.figure(figsize=(14, 7))
plt.plot(df['Volatility'], label='Annualized Volatility')
plt.title('Hyundai Motor Company Stock Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.show()


# In[21]:


# Display and plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[ ]:





# ## Time Series Analysis and Forecasting for Hyundai Stock Prices

# In[22]:



# Plot the time series
plt.figure(figsize=(10, 6))
plt.plot(df['Close'])
plt.title('Hyundai Stock Prices')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Perform ADF test
result = adfuller(df['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
for key, value in result[4].items():
    print('Critial Values:')
    print(f'   {key}, {value}')


# ### the data is not sationary 

# In[23]:


# Differencing to achieve stationarity
df['Close_diff'] = df['Close'].diff().dropna()

# Plot the differenced series
plt.figure(figsize=(10, 6))
plt.plot(df['Close_diff'])
plt.title('Differenced Hyundai Stock Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Close Price')
plt.show()

# Perform ADF test on the differenced series
result_diff = adfuller(df['Close_diff'].dropna())
print('ADF Statistic (differenced):', result_diff[0])
print('p-value (differenced):', result_diff[1])
for key, value in result_diff[4].items():
    print('Critical Values (differenced):')
    print(f'   {key}, {value}')


# In[24]:


# Plot ACF and PACF for the differenced series
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(df['Close_diff'].dropna(), lags=40, ax=ax[0])
plot_pacf(df['Close_diff'].dropna(), lags=40, ax=ax[1])
plt.show()


# In[25]:




# Fit the model using pmdarima's auto_arima
auto_model = pm.auto_arima(df['Close'], seasonal=False, stepwise=True, trace=True)

# Print the summary of the auto_arima model
print(auto_model.summary())


# In[26]:



# Extract the optimal order from auto_arima
order = auto_model.order
print('Optimal order:', order)

# Fit the ARIMA model
model = ARIMA(df['Close'], order=order)
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())


# In[27]:


# Forecasting
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, closed='right')

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('Hyundai Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[28]:


# Forecasting with confidence intervals
forecast_steps = 30
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int()

# Create a date index for the forecast period
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps+1, closed='right')

# Plot the actual and forecasted values with confidence intervals
plt.figure(figsize=(12, 8))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.title('Hyundai Stock Price Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Print the forecasted values with confidence intervals
forecast_df = pd.DataFrame({
    'Forecast': forecast,
    'Lower CI': confidence_intervals.iloc[:, 0],
    'Upper CI': confidence_intervals.iloc[:, 1]
}, index=forecast_index)

print(forecast_df)


# In[29]:


# Check forecast result
forecast_steps = 30
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int()

# Debugging: print the forecast results and confidence intervals
print(f"Forecast: {forecast}")
print(f"Confidence Intervals: {confidence_intervals}")

# Ensure the forecast index is correctly aligned
last_date = df.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

# Debugging: print the forecast index
print(f"Forecast Index: {forecast_index}")

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'Forecast': forecast,
    'Lower CI': confidence_intervals.iloc[:, 0],
    'Upper CI': confidence_intervals.iloc[:, 1]
}, index=forecast_index)

# Debugging: print the forecast DataFrame
print(forecast_df)

# Plot the actual and forecasted values with confidence intervals
plt.figure(figsize=(12, 8))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.title('Hyundai Stock Price Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[30]:


from statsmodels.tsa.arima.model import ARIMA

# Forecasting with confidence intervals
forecast_steps = 30
forecast_result = model_fit.get_forecast(steps=forecast_steps)
forecast = forecast_result.predicted_mean
confidence_intervals = forecast_result.conf_int()

# Create a date index for the forecast period
last_date = df.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B')

# Create a DataFrame for the forecast using the forecast_index
forecast_df = pd.DataFrame({
    'Forecast': forecast.values,
    'Lower CI': confidence_intervals.iloc[:, 0].values,
    'Upper CI': confidence_intervals.iloc[:, 1].values
}, index=forecast_index)

# Debugging: print the forecast DataFrame
print(forecast_df)

# Plot the actual and forecasted values with confidence intervals
plt.figure(figsize=(12, 8))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 confidence_intervals.iloc[:, 0], 
                 confidence_intervals.iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.title('Hyundai Stock Price Forecast with Confidence Intervals')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[31]:


print(forecast_df.head())


# In[32]:


model_fit.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[33]:


# Plot residuals
residuals = model_fit.resid

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of the ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.show()

# Plot ACF of residuals
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
plot_acf(residuals, lags=40, ax=ax)
plt.title('ACF of Residuals')
plt.show()


# In[ ]:





# ## Hyundai Stock Price Prediction Using Linear Regression and Random Forest

# In[34]:


df2


# In[35]:


# Define the features (X) and the target (y)
X = df2.drop('Close', axis=1)  # Assuming 'Close' is the column with stock prices
y = df2['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[36]:


# Initialize and fit the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_lr = linear_regressor.predict(X_test)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f'Linear Regression Mean Squared Error: {mse_lr}')
print(f'Linear Regression R^2 Score: {r2_lr}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.7)
plt.xlabel('Actual Stock Prices')
plt.ylabel('Predicted Stock Prices')
plt.title('Linear Regression: Actual vs Predicted Stock Prices')
plt.show()


# In[37]:


# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate the Random Forest Regression model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Regression Mean Squared Error: {mse_rf}')
print(f'Random Forest Regression R^2 Score: {r2_rf}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.7)
plt.xlabel('Actual Stock Prices')
plt.ylabel('Predicted Stock Prices')
plt.title('Random Forest Regression: Actual vs Predicted Stock Prices')
plt.show()


# In[38]:


print(f'Linear Regression Mean Squared Error: {mse_lr}')
print(f'Linear Regression R^2 Score: {r2_lr}')
print(f'Random Forest Regression Mean Squared Error: {mse_rf}')
print(f'Random Forest Regression R^2 Score: {r2_rf}')


# In[ ]:





# ### Fine-Tuning the Random Forest Model

# In[39]:


# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of random samples
    cv=3,  # Number of cross-validation folds
    n_jobs=-1,  # Use all available cores
    verbose=2,
    random_state=42
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = random_search.best_params_
best_rf_model = random_search.best_estimator_

print(f'Best parameters found: {best_params}')


# ### Update Parameters

# In[40]:



# Update the best parameters if necessary
best_rf_model = RandomForestRegressor(
    n_estimators=200,
    max_features='sqrt',
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the model on the training data
best_rf_model.fit(X_train, y_train)

# Predict on the test data
y_pred = best_rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error of the best Random Forest model: {mse}')


# In[41]:


# Get feature importances
importances = best_rf_model.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances in Random Forest Model')
plt.show()


# ### Hyperparameter Optimization of Random Forest Regressor using Hyperopt

# In[42]:



def objective(params):
    rf = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_features=params['max_features'],
        max_depth=params['max_depth'] if params['max_depth'] is not None else None,
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        random_state=42
    )
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Define the search space
space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'max_features': hp.choice('max_features', ['auto', 'sqrt', 'log2']),
    'max_depth': hp.choice('max_depth', [None, 10, 20, 30]),
    'min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4])
}

# Initialize Trials object to store search results
trials = Trials()

# Run Hyperopt optimization
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,  # Number of evaluations
            trials=trials,
            rstate=np.random.default_rng(seed=42))

print(f'Best parameters found: {best}')


# In[43]:


param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}


# In[44]:



# Define the model
rf = RandomForestRegressor()

# Define the parameter grid
param_dist = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Define the search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, verbose=1, random_state=42, n_jobs=-1)

# Fit the search
random_search.fit(X_train, y_train)

# Print best parameters and score
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best score found: {random_search.best_score_}")


# In[45]:


# Define the final model with the best parameters
best_rf = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='log2',
    max_depth=30,
    random_state=42
)

# Train the model
best_rf.fit(X_train, y_train)

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on the test set: {mse}")


# In[46]:




# Convert the trials to a DataFrame
results = pd.DataFrame(trials.results)
print(f'Best hyperparameters found: {best}')
print(f'All trials: \n{results}')


# In[47]:


# Extract best hyperparameters
best_params = {
    'n_estimators': [100, 200, 300, 400][best['n_estimators']],
    'max_features': ['auto', 'sqrt', 'log2'][best['max_features']],
    'max_depth': [None, 10, 20, 30][best['max_depth']],
    'min_samples_split': [2, 5, 10][best['min_samples_split']],
    'min_samples_leaf': [1, 2, 4][best['min_samples_leaf']]
}

# Train the final model
final_model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
final_model.fit(X_train, y_train)


# In[48]:



# Predictions on the test set
final_predictions = final_model.predict(X_test)

# Calculate the Mean Squared Error
final_mse = mean_squared_error(y_test, final_predictions)
print(f'Mean Squared Error of the final model: {final_mse}')


# In[49]:



# Save the model
joblib.dump(final_model, 'final_random_forest_model.pkl')


# In[ ]:




