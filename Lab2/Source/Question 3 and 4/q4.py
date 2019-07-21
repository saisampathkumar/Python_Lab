from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, SpatialDropout1D, SimpleRNN
from keras.callbacks import ModelCheckpoint,TensorBoard
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

spam= pd.read_csv('C:\Courses_Masters\Python and DL programming\lab2\q3\SPAM text message 20170820 - Data.csv', encoding='latin-1')
spam.head(2)
data = spam[['Category','Message']]

data['Message'] = data['Message'].apply(lambda x: x.lower())
data['Message'] = data['Message'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

max_features = 2000
tokenizer = Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(data['Message'].values)
X = tokenizer.texts_to_sequences(data['Message'].values)
print("tokenizer output texts_to_sequences: ",X)
X = pad_sequences(X)
print("\n pad_sequences output: \n",X)
batch_size = 128
epochs = 10
embed_dim = 128
lstm_out = 196

# num_classes=data.shape[1]
# print("num_classes:",num_classes)

model = Sequential()
# model.add(Embedding(20000, 100, input_length=56))
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='sigmoid'))
# model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

Y = pd.get_dummies(data['Category']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 42)

X_val = X_train[:500]
Y_val = Y_train[:500]
partial_X_train = X_train[500:]
partial_Y_train = Y_train[500:]
batch_size = 512
# Fit the model
tensorboard = TensorBoard(log_dir="logs/{}",histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(partial_X_train,partial_Y_train, validation_data=(X_val, Y_val), batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[tensorboard])

# # Evaluate the model
# score, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
#
# print('Score: %.2f' %(score))
# print('Accuracy: %.2f' % (accuracy))