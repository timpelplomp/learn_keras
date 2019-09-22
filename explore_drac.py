from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import json
import math


# from keras.models import load_model

with open("total_train_sets/enc_sents.json", "r", encoding="utf-8") as f:
    enc_sents = json.load(f)
with open("total_train_sets/cl_sents.json", "r", encoding="utf-8") as d:
    cl_sents = json.load(d)

sent_amount = math.floor(round(len(enc_sents) / 2))

x_train = enc_sents[:sent_amount]
x_test = enc_sents[sent_amount:]

y_train = cl_sents[:sent_amount]
y_test = cl_sents[sent_amount:]

max_features = 20000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print("x_train imdb: ")
print(x_train)
print("y_train imdb: ")
print(y_test)

print("x_test imdb: ")
print(x_test)
print("y_test imdb: ")
print(y_test)


print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
# Test score: 1.0290148921877145
# Test accuracy: 0.8094
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)