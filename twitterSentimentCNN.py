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

cols = ['sentiment','id','date','query_string','user','text']
dfTrain = pd.read_csv("./dataset/trainingData.csv",header=None, names=cols, encoding='ISO-8859-1')
dfTrain.drop(['id','date','query_string','user'],axis=1,inplace=True)

# Shuffle the data
dfTrain.sample(frac=1)

dfTest = pd.read_csv("./dataset/testData.csv",header=None, names=cols, encoding='ISO-8859-1')
dfTest.drop(['id','date','query_string','user'],axis=1,inplace=True)

dfTest.sample(frac=1)

dfTest = dfTest[dfTest.sentiment != 2]

y_train = dfTrain['sentiment'].tolist()
x_train = dfTrain['text'].tolist()

y_test = dfTest['sentiment'].tolist()
x_test = dfTest['text'].tolist()

# Only take the front and back 160000 samples
# x_train = x_train[:160000] + x_train[-160000:]
# y_train = y_train[:160000] + y_train[-160000:]

# x_test = x_train[80000:88000] + x_train[-88000:-80000]
# y_test = y_train[80000:88000] + y_train[-88000:-80000]

for index, sentiment in enumerate(y_train):
    if sentiment == 4:
        y_train[index] = 1

for index, sentiment in enumerate(y_test):
    if sentiment == 4:
        y_test[index] = 1

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)

x_test = tokenizer.texts_to_sequences(x_test)

# Import statements for various advanced activation functions we tried
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU

x_train = sequence.pad_sequences(x_train, maxlen=45)
x_test = sequence.pad_sequences(x_test, maxlen=45)

model = Sequential()

model.add(Embedding(100000, 64, input_length=45))

# Activation function
leakyrelu = LeakyReLU(alpha=0.1)

model.add(Conv1D(64, 2, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

model.add(Conv1D(32, 3, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

model.add(Conv1D(16, 4, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

model.add(Conv1D(8, 4, activation="relu", padding="same"))
model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))

##### Good config: 320000 training samples, rmsprop, 3 conv layers, relu

# Flattens the CNN so we can add a standard densely-connected network layer
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

# Using tensorboard we can log the results of our training and provide graphs in our report
tb = TensorBoard(log_dir='./logs', batch_size=64, write_graph=True, write_grads=True, write_images=True)

# Compile our model using binary crossentropy as our loss function, and the rmsprop optimizer.
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Time to train!
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
