"""
Time Series Forecasting Tutorial

reference: https://engineering.99x.io/time-series-forecasting-in-machine-learning-3972f7a7a467
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('data/PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)

df.plot(style='.', figsize=(15,5), color=color_pal[0], title='PJME Energy Use in MW')
plt.show()


## Train test split
train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

# plotting the data
fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()

# general idea about one week of data
df.loc[(df.index > '2010-01-01') & (df.index < '2010-01-08')].plot(
    figsize=(15, 5), title='Week of Data')
plt.show()


## Feature Creation
def create_features(df):
    """
    Creates time series features based on time series index
    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    df : pandas dataframe

    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['quarter'] = df.index.quarter
    
    df['month'] = df.index.month
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    return df

df = create_features(df)


## Visualize our feature target relationship (hourly consumption)
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x='hour', y='PJME_MW')
ax.set_title('MW by Hour')

# Visualize our feature target relationship (monthly consumption)
fig, ax = plt.subplots(figsize=(10,8))
sns.boxplot(data=df, x='month', y='PJME_MW')
ax.set_title('MW by Month')

## Create our model
train = create_features(train)
test = create_features(test)

FEATURES = ['hour', 'dayofweek', 'quarter', 'month', 'year',
       'dayofyear']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50,
                       learning_rate=0.01)
reg.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)],
        verbose=100)


## Feature Importances
fi = pd.DataFrame(reg.feature_importances_,
             index=reg.feature_names_in_,
             columns=['importance'])

fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()


## Forecast on Test
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

# plot the forecast
ax = df[['PJME_MW']].plot(figsize=(15,5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Predictions')
plt.show()

# plotting both raw data and predictions on one week of data
ax = df.loc[(df.index > '2018-04-01') & (df.index < '2018-04-08')]['PJME_MW'].plot(
    figsize=(15, 5), title='Week of Data')
df.loc[(df.index > '2018-04-01') & (df.index < '2018-04-08')]['prediction'].plot(
    style='.')
plt.legend(['Truth Data', 'Prediction'])
plt.show()


score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
print(f'RMSE Score on test set: {score:0.2f}')


## Calculate Error
#  Looking the worst and best predicted days

test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date

# worst predictions
test.groupby('date')['error'].mean().sort_values(ascending=False).head(5)

# best predictions
test.groupby('date')['error'].mean().sort_values(ascending=True).head(5)
