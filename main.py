from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import twitter
from birdy.twitter import StreamClient

# Setup the tokenizer so the computer has knowledge of words
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
    return tokenizer

# Load the saved model that was trained from twitterSentimentCNN.py
def loadModel():
    print('Loading Prediction Model...')
    # Load json and create model
    jsonFile = open('model.json', 'r')
    loadedJsonModel = jsonFile.read()
    jsonFile.close()
    loadedModel = model_from_json(loadedJsonModel)

    # Load weights into new model
    loadedModel.load_weights("model.h5")
    print("Loaded Model from Disk!")

    # Compile the loaded model for use
    loadedModel.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loadedModel

# Setup the connection to the Twitter web api, can get previously tweeted tweets or stream live tweets
def setUpTwitterConnection(stream=True):
    api = twitter.Api(consumer_key='wZFb3lcFyezEJuIGUMUvtZGwJ',
                  consumer_secret='xBiVivqFgEUpwrDSvT0JWCIOUssgktQtpqN41tsM1q9ErnAO3s',
                  access_token_key='982689083188285440-wIW0kURcR9rMSlB07cDuK59Z6c23pUH',
                  access_token_secret='nfmjnZxTyo79t4nMqr9xoQJ5lPKnQYxMVZnmkMUWMXpk1')

    streamApi = StreamClient('wZFb3lcFyezEJuIGUMUvtZGwJ',
                    'xBiVivqFgEUpwrDSvT0JWCIOUssgktQtpqN41tsM1q9ErnAO3s',
                    '982689083188285440-wIW0kURcR9rMSlB07cDuK59Z6c23pUH',
                    'nfmjnZxTyo79t4nMqr9xoQJ5lPKnQYxMVZnmkMUWMXpk1')

    if stream:
        return streamApi
    else:
        return api

def main():
    tokenizer = setUpTokenizer()
    model = loadModel()
    twitterApi = setUpTwitterConnection()

    while True:
        # Get relevant user input
        print('Twitter sentiment analizer ready, enter \'quit\' in topic selection to exit')
        topic = str(input('Select topic: '))
        if topic == 'quit':
            break
        numTweets = int(input('Select number of tweets to analyze: '))

        resource = twitterApi.stream.statuses.filter.post(track=topic)

        index = 0
        tweets = []
        # Keep getting live tweets on 'topic' until the users chosen tweet number is reached
        for data in resource.stream():
            if index >= numTweets:
                break
            if 'lang' in data:
                if data.lang == 'en':
                    index = index + 1
                    print(str(index) + '/' + str(numTweets))
                    tweets.append(data.text)

        # Prepare the gathered tweets for the CNN to predict, i.e turn the tweets into vectors
        tweets = tokenizer.texts_to_sequences(tweets)
        tweets = sequence.pad_sequences(tweets, maxlen=45)

        # Get the models predictions for each of the tweets
        predictions = model.predict(np.array(tweets))

        # Analize the predictions based on threshold values for negative, neutral, and positive sentiment
        negativePredictions = 0
        neutralPredictions = 0
        positivePredictions = 0
        for prediction in predictions:
            if prediction[0] < 0.50:
                negativePredictions = negativePredictions + 1
            elif prediction[0] >= 0.50 and prediction[0] <= 0.65:
                neutralPredictions = neutralPredictions + 1
            else:
                positivePredictions = positivePredictions + 1

        print('Results:')
        print(str((negativePredictions/numTweets) * 100) + '% Negative Sentiment')
        print(str((neutralPredictions/numTweets) * 100) + '% Neutral Sentiment')
        print(str((positivePredictions/numTweets) * 100) + '% Positive Sentiment')

main()
