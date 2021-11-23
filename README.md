
## SBI-10-year_stock_Prediction-And-Forecasting-Using-Stacked-LSTM

#### Stock-Market-Forecasting using DEEP LEARNING

Traditionally most machine learning (ML) models use as input features some observations (samples / examples) but there is no time dimension in the data.
Time-series forecasting models are the models that are capable to predict future values based on previously observed values. Time-series forecasting is widely used for non-stationary data. Non-stationary data are called the data whose statistical properties e.g. the mean and standard deviation are not constant over time but instead, these metrics vary over time.

These non-stationary input data (used as input to these models) are usually called time-series. Some examples of time-series include the temperature values over time, stock price over time, price of a house over time etc. So, the input is a signal (time-series) that is defined by observations taken sequentially in time.

#### A time series is a sequence of observations taken sequentially in time.

## The LSTM model

Long short-term memory (LSTM) is an artificial recurrent neural network (RNN) architecture used in the field of deep learning. Unlike standard feedforward neural networks, LSTM has feedback connections. It can not only process single data points (e.g. images), but also entire sequences of data (such as speech or video inputs).

## Imports/Initial Data

To begin our project, we import numpy for making scientific computations, pandas for loading and modifying datasets, and matplotlib for plotting graphs.

```bash
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
```

After making the necessary imports, we load data on State Bank of India past stock prices.

```bash
state_bank=pd.read_csv('SBIN.NS.csv')
```

### Incorporating Timesteps Into Data

We should input our data in the form of a 3D array to the LSTM model. First, we create data in 60 timesteps before using numpy to convert it into an array. Finally, we convert the data into a 3D array with X_train samples, 60 timestamps, and one feature at each step.

```bash
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
```
### Creating the LSTM Model

Before we can develop the LSTM, we have to make a few imports from Keras: Sequential for initializing the neural network, LSTM to add the LSTM layer, Dropout for preventing overfitting with dropout layers, and Dense to add a densely connected neural network layer.

```bash
from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dropout

from keras.layers import Dense
```

Specifying 0.2 in the Dropout layer means that 20% of the layers will be dropped. Following the LSTM and Dropout layers, we add the Dense layer that specifies an output of one unit. To compile our model we use the Adam optimizer and set the loss as the mean_squared_error. After that, we fit the model to run for 100 epochs (the epochs are the number of times the learning algorithm will work through the entire training set) with a batch size of 32.

```bash
regressor = Sequential()

regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))
```
### Making Predictions on the Test Set

#### We start off by importing the test set

```bash
dataset_train = state_bank_stock.iloc[:1976,2:3]

data_testing = state_bank_stock.iloc[1976:,2:3]
```
Before predicting future stock prices, we have to modify the test set (notice similarities to the edits we made to the training set): merge the training set and the test set on the 0 axis, set 60 as the time step again, use MinMaxScaler, and reshape data. Then, inverse_transform puts the stock prices in a normal readable format.

```bash
total_dataset = pd.concat((dataset_train,data_testing),axis=0)

input_to_model = total_dataset[len(total_dataset)-len(data_testing)-60:].values

inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []

for i in range(60,545):

X_test.append(inputs[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)
```

### Plotting the Results

After all these steps, we can use matplotlib to visualize the result of our predicted stock price and the actual stock price.

```bash
plt.figure(figsize=(15,6))

plt.plot(real_stock_price, color = 'red', label = 'Real state bank of india Stock Price')

plt.plot(predicted_stock_price2, color = 'blue', label = 'predicted_state bank of india_stock_price')

plt.title('state bank of india 10 year Stock Price Prediction')

plt.xlabel('Time - 2020-12-11--2010-12-14')

plt.ylabel('state bank of india Stock Price')

plt.legend()

plt.show()
```
![Alt text](https://github.com/sonu275981/SBI-10-year_stock_Prediction-And-Forecasting-Using-Stacked-LSTM/blob/795b113a2553ce3f4d01757d0fdc63f0a1902c2b/Demo.png?raw=true "Face-Recognition-Attendance-System")

While the exact price points from our predicted price weren’t always close to the actual price, our model did still indicate overall trends such as going up or down. This project teaches us the LSTMs can be somewhat effective in times series forecasting.



