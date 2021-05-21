import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random

#Load network model
lstm_model = tf.keras.models.load_model('multistep_model')

#Read csv file
df = pd.read_csv(
    'SensorValues.csv',
    parse_dates=['Time'],
    index_col='Time'
)
#Scale values
scaler = MinMaxScaler(feature_range=(-1,1))
test_size = int(len(df)*0.9)
value = scaler.fit_transform(df[test_size:].values)

history_size = 6*24*2 #The history size, size of the window
hours = 1 #How many hours to predict, should be the same as in main.py
target_size = 6*hours

#Create input window
history = []
index = random.randint(history_size, value.shape[0]-target_size)
history.append(value[range(index-history_size, index)])
history = np.array(history)
target = value[index+1:index+target_size+1]

#Predict values
yhat = lstm_model.predict(history)
yhat = yhat[0]

#Inverse scale back to original sizes.
yhat = scaler.inverse_transform(yhat)
target = scaler.inverse_transform(target)

#Plot all values in graphs
font = {'family': 'normal', 'weight' : 'bold', 'size' : 15}
plt.rc('font', **font)
fig, axs = plt.subplots(3,2)
axs[0,0].plot(target[:,0], color='blue', label='Actual value')
axs[0,0].plot(list(range(target_size)),yhat[:,0], 'r+', label='Predicted value')
axs[0,0].set_title('Temperature')
axs[0,0].set_ylim([19, 23])
axs[0,1].plot(target[:,1], color='blue', label='Actual value')
axs[0,1].plot(list(range(target_size)),yhat[:,1], 'r+', label='Predicted value')
axs[0,1].set_title('Humidity')
axs[0,1].set_ylim([18, 50])
axs[1,0].plot(target[:,2], color='blue', label='Actual value')
axs[1,0].plot(list(range(target_size)),yhat[:,2], 'r+', label='Predicted value')
axs[1,0].set_title('Pressure')
axs[1,0].set_ylim([942, 1000])
axs[1,1].plot(target[:,3], color='blue', label='Actual value')
axs[1,1].plot(list(range(target_size)),yhat[:,3], 'r+', label='Predicted value')
axs[1,1].set_title('gas/1000')
axs[1,1].set_ylim([8, 1700])
axs[2,0].plot(target[:,4], color='blue', label='Actual value')
axs[2,0].plot(list(range(target_size)),yhat[:,4], 'r+', label='Predicted value')
axs[2,0].set_title('Lux')
axs[2,0].set_ylim([-10, 75])
axs[2,1].plot(target[:,5], color='blue', label='Actual value')
axs[2,1].plot(list(range(target_size)),yhat[:,5], 'r+', label='Predicted value')
axs[2,1].set_title('co2')
axs[2,1].set_ylim([300, 1200])
plt.subplots_adjust(hspace=0.5)
plt.legend(loc='upper center', bbox_to_anchor=(-0.1, 4.5))
plt.show()