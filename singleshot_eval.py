import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

#Read csv file
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)
#Scale values
batch_size = int(len(df)*0.9)
scaler = MinMaxScaler(feature_range=(-1,1))
value = scaler.fit_transform(df[batch_size:].values)

#Split values into x and y
x_val = value[:-1]
y_val = value[1:]

#Create prediction set
x, y = [], []
history_size = 6*24*2
target_size = 6
for i in range(history_size, len(value)-target_size):
    indices = range(i-history_size, i)
    x.append(x_val[indices])
    y.append(y_val[i:i+target_size])
x_val, y_val = np.asarray(x), np.asarray(y)

#Load LSTM model
model = tf.keras.models.load_model('multistep_model')
###########################MEASURE PERFORMANCE###########################
times = []
temp_error = []
hum_error = []
press_error = []
gas_error = []
lux_error = []
co_error = []

#invesrse scale of observed values
for i in range(y_val.shape[0]):
    y_val[i, :, :] = scaler.inverse_transform(y_val[i,:,:])

#Predict all values and store absolute error and prediction time
for i in range(len(x_val)):
    x_in = x_val[i] #Select single input for single output
    x_in = x_in.reshape(1, x_in.shape[0], x_in.shape[1]) #Reshape to 3-D for input
    t0 = time.time() #Start time
    yhat = model.predict(x_in) #Predict
    t1 = time.time() #Stop time
    times.append(t1-t0) #Store time it took to predict
    yhat[0] = scaler.inverse_transform(yhat[0]) #Inverse scale predicted values
    #Calculate the errors of the predictions
    for j in range(yhat.shape[1]):
        temp_error.append(abs(y_val[i,j,0]-yhat[0,j,0]))
        hum_error.append(abs(y_val[i,j,1]-yhat[0,j,1]))
        press_error.append(abs(y_val[i,j,2]-yhat[0,j,2]))
        gas_error.append(abs(y_val[i,j,3]-yhat[0,j,3]))
        lux_error.append(abs(y_val[i,j,4]-yhat[0,j,4]))
        co_error.append(abs(y_val[i,j,5]-yhat[0,j,5]))

#Calculate mean of all errors
temp_mean = np.mean(temp_error)
hum_mean = np.mean(hum_error)
press_mean = np.mean(press_error)
gas_mean = np.mean(gas_error)
lux_mean = np.mean(lux_error)
co_mean = np.mean(co_error)

#Calculate standard deviation of all errors
temp_std = np.std(temp_error)
hum_std = np.std(hum_error)
press_std = np.std(press_error)
gas_std = np.std(gas_error)
lux_std = np.std(lux_error)
co_std = np.std(co_error)

#Calculate average time and std to produce a prediction
time_mean = np.mean(times)
time_std = np.std(times)

#Print all values
print(f'Time mean={time_mean} std={time_std}')
print(f'Temp mean={temp_mean} std={temp_std}')
print(f'Hum mean={hum_mean} std={hum_std}')
print(f'Press mean={press_mean} std={press_std}')
print(f'Gas mean={gas_mean} std={gas_std}')
print(f'Lux mean={lux_mean} std={lux_std}')
print(f'co2 mean={co_mean} std={co_std}')