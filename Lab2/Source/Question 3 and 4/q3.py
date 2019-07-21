import tensorboard as tensorboard
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten, LSTM, SpatialDropout1D, SimpleRNN
from keras.layers.convolutional import Conv1D
from keras.constraints import maxnorm
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Loading dataset
spam= pd.read_csv('C:\Courses_Masters\Python and DL programming\lab2\q3\SPAM text message 20170820 - Data.csv', encoding='latin-1')
spam.head(2)
data = spam[['Category','Message']]
#formatting data

data['Message'] = data['Message'].apply(lambda x: x.lower()) #lower casing the text
data['Message'] = data['Message'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['Message'].values) #tokenizing
X = tokenizer.texts_to_sequences(data['Message'].values)
print("tokenizer output texts_to_sequences: ",X)
X = pad_sequences(X)
print("\n pad_sequences output: \n",X)
embed_dim = 128
lstm_out = 196
Y = pd.get_dummies(data['Category']).values
#splitting testing and training data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)
#selecting data for training and validation
X_val = X_train[:1500]
Y_val = Y_train[:1500]
partial_X_train = X_train[1500:]
partial_Y_train = Y_train[1500:]
batch_size = 512
num_classes = Y_test.shape[1]
#creating model
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X_train.shape[1]))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy']) #compiling model
print(model.summary())
#fitting into model
#tensor board graph generation.
tensorboard = TensorBoard(log_dir="logs2/{}",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(partial_X_train,partial_Y_train,epochs = 10,batch_size=batch_size,validation_data=(X_val, Y_val),callbacks=[tensorboard])
