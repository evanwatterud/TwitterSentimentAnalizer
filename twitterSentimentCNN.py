from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.layers import Dense, LSTM, Conv1D, Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.layers import MaxPooling1D
from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd

# Parse the training set csv and remove the columns that aren't useful
cols = ['sentiment','id','date','query_string','user','text']
dfTrain = pd.read_csv("./dataset/trainingData.csv", header=None, names=cols, encoding='ISO-8859-1')
dfTrain.drop(['id','date','query_string','user'], axis=1, inplace=True)

# Shuffle the data
dfTrain.sample(frac=1)

# Parse the testing set csv and remove the columns that aren't useful
dfTest = pd.read_csv("./dataset/testData.csv",header=None, names=cols, encoding='ISO-8859-1')
dfTest.drop(['id','date','query_string','user'],axis=1,inplace=True)

dfTest.sample(frac=1)

# Remove the test set rows that have neutral sentiment
dfTest = dfTest[dfTest.sentiment != 2]

y_train = dfTrain['sentiment'].tolist()
x_train = dfTrain['text'].tolist()

y_test = dfTest['sentiment'].tolist()
x_test = dfTest['text'].tolist()

# Positive sentiment is labeled with 4s for some reason, changing them to 1s, negative sentiment is 0
for index, sentiment in enumerate(y_train):
    if sentiment == 4:
        y_train[index] = 1

for index, sentiment in enumerate(y_test):
    if sentiment == 4:
        y_test[index] = 1

# Setup the tokenizer to learn the top 100000 words from the training set, may be too high but it works for now
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

# Import statements for various advanced activation functions we tried
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

# Pad the tweets so that they're less than 45 words long, they generally shouldn't be longer than this anyways
x_train = sequence.pad_sequences(x_train, maxlen=45)
x_test = sequence.pad_sequences(x_test, maxlen=45)

model = Sequential()

# Add the word embedder, creates a vector space where words that are similar are close together in the vector space
# This is what gives the model it's 'understanding' of words
model.add(Embedding(100000, 64, input_length=45))

# Additional activation function, haven't tested any activation functions other than normal relu yet though
leakyrelu = LeakyReLU(alpha=0.1)

model.add(Conv1D(64, 2, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

model.add(Conv1D(16, 4, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

model.add(Conv1D(8, 4, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

# Flattens the CNN so we can add a standard densely-connected network layer
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Rmsprop found to be the best optimizer for this problem with the current hyperparameters
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model, no more than 2 epochs needed given the size of the training set
model.fit(np.array(x_train), np.array(y_train), epochs=2, validation_data=(np.array(x_test), np.array(y_test)), callbacks=[tb], batch_size=64)

# Evaluate the model against the testing set
evaluation = model.evaluate(np.array(x_test), np.array(y_test))

print("The accuracy of the model is: %.2f%%" % (evaluation[1]*100))

# Serialize model to JSON
jsonModel = model.to_json()
with open("model.json", "w") as jsonFile:
    jsonFile.write(jsonModel)
# Serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# Load json and create model
jsonFile = open('model.json', 'r')
loadedJsonModel = jsonFile.read()
json_file.close()
loadedModel = model_from_json(loadedJsonModel)

# Load weights into new model
loadedModel.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loadedModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
evaluation = loadedModel.evaluate(np.array(x_test), np.array(y_test))

print("The accuracy of the model is: %.2f%%" % (evaluation[1]*100))
