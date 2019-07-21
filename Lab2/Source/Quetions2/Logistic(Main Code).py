# Logistic regression with learning rate, batch size and Optimizer
# importing some functions
from __future__ import print_function
import numpy as np
import pandas as pd
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# saving the event in a certain location to project the graphs
tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0,write_graph=True, write_images=True)

# Reading csv file
df = pd.read_csv('Modified_Rent_Data.csv')
data = pd.DataFrame(df, columns=['price', 'price_aprox_local_currency', 'price_aprox_usd', 'surface_total_in_m2',
                                    'surface_covered_in_m2', 'price_usd_per_m2',])
label_col = 'price'
# Splitting the available data for train set and test set
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 0:3], data.iloc[:, 3],
                                                                  test_size=0.5, random_state=87)
np.random.seed(155)

# converting the available data to numpy arrays
dfs = x_train.append(x_test)
minimum = np.min(dfs)
maximum = np.max(dfs)
mu = np.mean(dfs)
sigma = np.std(dfs)

df2 = pd.DataFrame()
for c in x_train.columns:
    df2[c] = (x_train[c] - mu[c]) / sigma[c]
x_train = np.array(df2)
y_train = np.array(y_train)

df1 = pd.DataFrame()
for c in x_test.columns:
    df1[c] = (x_test[c] - mu[c]) / sigma[c]
x_test = np.array(df1)
y_test = np.array(y_train)

# Printing the shapes of the arrays
print('\n X Input shape:', x_train.shape)
print('Y Input shape:', y_train.shape)
print('Train samples: ', x_train.shape[0])
print('Test samples: ', x_test.shape[0],'\n')

# Creating a Logistic model
Lmodel = Sequential()
# provided sigmoid function as an activation function for logistic regression
Lmodel.add(Dense(100, activation='sigmoid', input_shape=(x_train.shape[1],)))
Lmodel.add(Dense(50, activation="sigmoid"))
Lmodel.add(Dense(1))
# We use binary cross entropy loss and rmsprop as optimizer for the logistic regression
Lmodel.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=[metrics.mae])
Lmodel.summary()

# Defining different types of epochs and batch sizes
# Fitting the models and saving the tensorboard events
epochs = 20
batch_size = 128
history = Lmodel.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2,validation_data=(x_test, y_test),callbacks=[tbCallBack])

train_score = Lmodel.evaluate(x_train, y_train, verbose=0)
test_score = Lmodel.evaluate(x_test, y_test, verbose=0)

# printing the loss and error
print('Train Mean Absolute Error : ', round(train_score[1], 1),
      '\n Train Loss: ', round(train_score[0], 1))
print('Val Mean Absolute Error : ', round(test_score[1], 1),
      '\n Val Loss: ', round(test_score[0], 1))


epochs = 50
batch_size = 128
history = Lmodel.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2,validation_data=(x_test, y_test),callbacks=[tbCallBack])

train_score = Lmodel.evaluate(x_train, y_train, verbose=0)
test_score = Lmodel.evaluate(x_test, y_test, verbose=0)

print('Train Mean Absolute Error : ', round(train_score[1], 1),
      '\n Train Loss: ', round(train_score[0], 1))
print('Val Mean Absolute Error : ', round(test_score[1], 1),
      '\n Val Loss: ', round(test_score[0], 1))

epochs = 50
batch_size = 256
history = Lmodel.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2,validation_data=(x_test, y_test),callbacks=[tbCallBack])

train_score = Lmodel.evaluate(x_train, y_train, verbose=0)
test_score = Lmodel.evaluate(x_test, y_test, verbose=0)

print('Train Mean Absolute Error : ', round(train_score[1], 1),
      '\n Train Loss: ', round(train_score[0], 1))
print('Val Mean Absolute Error : ', round(test_score[1], 1),
      '\n Test Loss: ', round(test_score[0], 1))