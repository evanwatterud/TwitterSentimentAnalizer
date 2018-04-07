from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

def setUpTokenizer():
    print('Setting Up Tokenizer...')

    cols = ['sentiment','id','date','query_string','user','text']
    dfTrain = pd.read_csv("./dataset/trainingData.csv",header=None, names=cols, encoding='ISO-8859-1')
    dfTrain.drop(['id','date','query_string','user'],axis=1,inplace=True)

    # Shuffle the data
    dfTrain.sample(frac=1)

    x_train = dfTrain['text'].tolist()

    tokenizer = Tokenizer(num_words=100000)
    tokenizer.fit_on_texts(x_train)

    print('Setup Complete!')

def loadModel():
    print('Loading Prediction Model...')
    # Load json and create model
    jsonFile = open('model.json', 'r')
    loadedJsonModel = jsonFile.read()
    json_file.close()
    loadedModel = model_from_json(loadedJsonModel)

    # Load weights into new model
    loadedModel.load_weights("model.h5")
    print("Loaded Model from Disk!")

    # Compile the loaded model for use
    loadedModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loadedModel

def main():
    setUpTokenizer()
    model = loadModel()
