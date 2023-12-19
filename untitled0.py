import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

store_sales = pd.read_csv("train.csv")

"""Check for null values in the dataset

"""



store_sales = store_sales.drop(['store', 'item'], axis=1)

store_sales['date'] = pd.to_datetime(store_sales['date'])

store_sales['date'] = store_sales['date'].dt.to_period("M")
monthly_sales = store_sales.groupby('date').sum().reset_index()

"""convert the resulting date column to timestamp datatype

"""

monthly_sales['date'] = monthly_sales['date'].dt.to_timestamp()

"""Visualization

"""

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Monthly income sales")
plt.show()

"""call the difference on the sales columns to make the sales data stationary"""

monthly_sales['sales_diff'] = monthly_sales['sales'].diff()
monthly_sales = monthly_sales.dropna()

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'], monthly_sales['sales'])
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Monthly sales difference")
plt.show()

"""Dropping off sales and date"""

supervised_data = monthly_sales.drop(['date','sales'], axis=1)

"""Preparing the supervised data"""

for i in range(1,11):
  col_name= 'month' + str(i)
  supervised_data[col_name] = supervised_data['sales_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop=True)

"""Split the data into Train and Test"""

train_data = supervised_data[:-12]
test_data = supervised_data[-12:]

scaler =  MinMaxScaler(feature_range=(-1,11))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

x_train = train_data[:,1:]
y_train = train_data[:,0:1]
x_test, y_test = test_data[:,1:], test_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()

"""Make prediction data frame to merge the predicted sales prices of all trained algs

"""

sales_dates = monthly_sales['date'][-12:].reset_index(drop=True)
predict_df = pd.DataFrame(sales_dates)

act_sales = monthly_sales['sales'][-13:].to_list()

"""To create the linear regression model and predicted output

"""

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
lr_pre = lr_model.predict(x_test)

lr_pre = lr_pre.reshape(-1,1)
lr_pre_test_set = np.concatenate([lr_pre, x_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

result_list = []
for index in range(0, len(lr_pre_test_set)):
  result_list.append(lr_pre_test_set[index][0] + act_sales[index])
lr_pre_series = pd.Series(result_list, name="Linear Prediction")
predict_df = predict_df.merge(lr_pre_series, left_index = True, right_index = True)
print(predict_df.columns)

# Confirm 'Linear Prediction' column exists in the DataFrame
if 'Linear Prediction' in predict_df.columns:
    # Calculate RMSE
    lr_mse = np.sqrt(mean_squared_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:]))
    print("Linear Regression RMSE:", lr_mse)
    lr_mae = mean_absolute_error(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
    print("Linear Regression MAE:", lr_mae)
    lr_r2 = r2_score(predict_df['Linear Prediction'], monthly_sales['sales'][-12:])
    print("Linear Regression R2:", lr_r2)
else:
    print("Column 'Linear Prediction' not found in predict_df.")

"""Visualization of the prediction against eh actual sales"""

#plt.figure(figsize=(15,5))
#Actual sales
#plt.plot(monthly_sales['date'], #predict_df['Linear Prediction'])

#Predicted sales
#plt.plot(predict_df['date'], predict_df['Linear Prediction'])
#plt.title("Customer sales Forecast using LR model")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.legend(['Actual Sales', 'Predicted Sales'])
#plt.show()

plt.figure(figsize=(15, 5))

# Actual sales
plt.plot(monthly_sales['date'], monthly_sales['sales'], label='Actual Sales')

# Predicted sales
plt.plot(monthly_sales['date'][-12:], predict_df['Linear Prediction'], label='Predicted Sales')

plt.title("Customer Sales Forecast using LR model")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

# Create a DataFrame with the predicted sales
predict_df = pd.DataFrame(index=monthly_sales['date'][-12:])
predict_df['Linear Prediction'] = lr_pre_test_set[:, 0]

# Convert the DataFrame to the desired format
predicted_sales_list = []
for index, row in predict_df.iterrows():
    predicted_sales_list.append({'date': index.strftime('%d/%m/%y'), 'sales': row['Linear Prediction']})