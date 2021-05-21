import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

#Read csv file
df = pd.read_csv(
    'SensorValues.csv', #Name of file
    parse_dates=['Time'], #date & time columns
    index_col='Time' #Which column should be the index
)
#Split intro train and test
val_size = int(len(df) * 0.7)
test_size = int(len(df) * 0.9)
train, test = df.iloc[:val_size].values, df.iloc[val_size:test_size].values

#Scale to -1 and 1
scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(df.values)
train = scaler.transform(train)
test = scaler.transform(test)

#Split into input and output
x_train = train[:-1]
y_train = train[1:]
x_test = test[:-1]
y_test = test[1:]

#Create moving window datasets
#Training batch
x, y = [], []
hours = 1 #How many hours to predict
history_size = 6*24*3 #history for input (3 days)
target_size = 6*hour #Number of timesteps to predict
for i in range(history_size, len(train)-target_size):
    indices = range(i-history_size, i)
    x.append(x_train[indices])
    y.append(y_train[i:i+target_size])
x_train, y_train = np.asarray(x), np.asarray(y)
train_batch = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_batch = train_batch.batch(32).repeat()

#Validation batch
x, y = [], []
for i in range(history_size, len(test)-target_size):
    indices = range(i-history_size, i)
    x.append(x_test[indices])
    y.append(y_test[i:i+target_size])
x_test, y_test = np.asarray(x), np.asarray(y)
val_batch = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_batch = val_batch.batch(32).repeat()

#Define LSTM model
n_features = 6 #number of features being predicted
multi_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(units=target_size*n_features),
    tf.keras.layers.Reshape([target_size, n_features])
])
#Compile model
multi_model.compile(loss='mean_squared_error', optimizer='adam')
history = multi_model.fit(
    train_batch,
    epochs=100,
    steps_per_epoch=1000,
    batch_size=32,
    validation_data=val_batch,
    validation_steps=100,
    shuffle=True
)
#Save model
multi_model.save('multistep_model')