# Title: Next Day Stock Forcaster
# Author: Ivor Myers
# GitHub: https://github.com/ivormyers
# 
# Desc: Machine learning python project that integrates the neural network capabilities of Tensorflow and Keras libraries. 
#Trained 50-neuron long short-term model (LSTM) on historical stock data to predict the next day’s stock price. 
#Utilized sliding window technique for data preparation and graphical visualization.  

# Bugs: None discovered
#
# Credit: NeuralNine on Youtube
# Assistance in developing code for neural network model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import timedelta 
import yfinance as yf
import pyarrow as py 

from sklearn.preprocessing import MinMaxScaler
    #support the machine learning nueral network
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#LOADING DATA Using YFinance
company = input('Predict Next Day of (TICKER): ')

print(f'Training model on last 4 years of {company} price data')

start = dt.datetime(2020,1,1)
end = dt.datetime.now()
data = yf.download(company,start,end)

#Prepare Data For Neural Network
    #scaling down prices to be inbetween 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
    #only concern on closing price not opening

prediction_days = 60
x_train = []
y_train = []

for x in range(prediction_days,len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x,0])
    y_train.append(scaled_data[x,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#Building The Model 

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(Dropout(.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(.2))
model.add(LSTM(units=50))
model.add(Dropout(.2))
model.add(Dense(units=1)) #prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Train the model and have early stopping system inplace
mfit = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

# Test The Model Accuracy #
test_start = dt.datetime(2023,1,1)
test_end = dt.datetime.now()
test_data = yf.download(company,test_start,test_end)
actual_prices = test_data['Close'].values
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values #looks right up to most recent date
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.transform(model_inputs)

#make prediciotns on test data
x_test = []

for x in range(prediction_days,len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#PREDICT THE NEXT DAY

real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs)]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    #use real data as input and then predict one next day
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Date: {test_end}")
print(f"Prediciton: {prediction}")
print(f"done!")

#PLOT TEST PREDICTIONS

epochret = early_stopping.stopped_epoch
if(epochret ==0): 
    epochret = 100
tomorrow = test_end + timedelta(days=1)


plt.figure(figsize=(10, 5)) 
plt.plot(mfit.history['loss'], label='Training Loss', color='blue')
plt.plot(mfit.history['val_loss'], label='Validation Loss', color='orange')
plt.scatter(epochret, 0, color='red', label=f'Stopped Epoch: {epochret}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Curves : With Early Stopping')


plt.figure(figsize=(10, 5))
plt.plot(test_data.index, actual_prices, color="black", label=f"Actual {company} Price")
plt.plot(test_data.index, predicted_prices, color="green", label=f"Predicted {company} Price")
plt.scatter(tomorrow, prediction, color='red', label=f'Tomorrow\' Prediction: {prediction}')
plt.title(f"{company} Share Price")
plt.xlabel(f"Date")
plt.ylabel(f"{company} Share Price")
plt.xlim(test_start, tomorrow)
plt.legend()
plt.show()




